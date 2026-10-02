"""Profile one Transformers LlamaAttention module with three backends.

Run on an idle physical GPU:
    python homework1/run.py --gpu 4

Each trace contains exactly one complete module forward after warmup.
No profiling markers are inserted into Transformers or individual operators.
"""

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

BACKENDS = ("eager", "sdpa", "flash_attention_2")
ROOT = Path(__file__).resolve().parents[1]


def nvidia_query(kind, fields):
    result = subprocess.run(
        ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    return list(csv.reader(result.stdout.strip().splitlines(), skipinitialspace=True))


def select_idle_gpu(requested):
    """Refuse a card with another compute process, even if its utilization is low."""
    busy = {row[0] for row in nvidia_query("compute-apps", "gpu_uuid,pid") if row}
    devices = nvidia_query("gpu", "index,uuid,name,memory.used,utilization.gpu")
    for index, uuid, name, memory, utilization in devices:
        if requested is not None and int(index) != requested:
            continue
        if uuid not in busy and int(memory) < 128 and int(utilization) == 0:
            return {
                "physical_index": int(index), "uuid": uuid, "name": name,
                "memory_used_before_mib": int(memory),
                "utilization_before_percent": int(utilization),
                "compute_processes_before": [],
            }
    raise RuntimeError("No requested GPU is idle. This script does not terminate processes.")


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def measure_forward(forward, repeats=30):
    """CUDA Events around complete eager module calls, outside the profiler."""
    import torch

    samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        start.record()
        output = forward()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
        del output
    return {
        "method": "CUDA Events around a complete module forward; no CUDA Graph",
        "median_us": statistics.median(samples),
        "min_us": min(samples), "max_us": max(samples), "samples_us": samples,
    }


def profile_forward(module, arguments, backend, destination):
    """One profiler and one outer annotation around the entire module call."""
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function

    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        with record_function(f"LlamaAttention.forward/{backend}"):
            output = module(**arguments)
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(destination))
    del output


def inspect_trace(path, backend):
    trace = json.loads(path.read_text())
    events = trace["traceEvents"]
    # PyTorch also projects the same scope onto the GPU lane. Count CPU calls only.
    outer = [
        event for event in events
        if event.get("name") == f"LlamaAttention.forward/{backend}"
        and event.get("cat") == "user_annotation" and event.get("ph") == "X"
    ]
    kernels = sorted(
        [event for event in events if event.get("cat") == "kernel" and event.get("ph") == "X"],
        key=lambda event: event["ts"],
    )
    operators = sorted({event["name"] for event in events if event.get("cat") == "cpu_op"})
    assert len(outer) == 1, "Expected exactly one complete forward annotation"
    assert kernels, "Profiler did not record CUDA kernels"
    assert all("grid" in event["args"] and "block" in event["args"] for event in kernels)
    if backend == "sdpa":
        assert "aten::scaled_dot_product_attention" in operators, operators
    elif backend == "flash_attention_2":
        assert any("flash_attn" in name for name in operators), operators
        assert "aten::scaled_dot_product_attention" not in operators, "Unexpected SDPA fallback"
        assert any("flash_fwd" in event["name"] for event in kernels), "No flash-attn forward kernel"
    else:
        assert "aten::scaled_dot_product_attention" not in operators, "Unexpected eager fallback"
        assert "aten::softmax" in operators, operators
    return {
        "trace": str(path.resolve()),
        "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "forward_annotations": len(outer),
        "kernel_count": len(kernels),
        "kernel_duration_sum_us": sum(event["dur"] for event in kernels),
        "operators": operators,
        "kernels": [
            {
                "name": event["name"],
                "start_us": event["ts"] - kernels[0]["ts"],
                "duration_us": event["dur"],
                "grid": event["args"]["grid"],
                "block": event["args"]["block"],
            }
            for event in kernels
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, help="Physical nvidia-smi index; default: first idle GPU")
    parser.add_argument("--out", type=Path, default=ROOT / "homework1/results/rtx5090")
    args = parser.parse_args()

    # Choose the device before importing libraries that may initialize CUDA.
    gpu = select_idle_gpu(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
    import torch
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding

    select_idle_gpu(gpu["physical_index"])
    seed = 20261001
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    args.out.mkdir(parents=True, exist_ok=True)

    # Synthetic weights; use the actual Transformers module without rewriting forward().
    config = LlamaConfig(
        hidden_size=1024, intermediate_size=4096, num_hidden_layers=1,
        num_attention_heads=16, num_key_value_heads=16, head_dim=64,
        max_position_embeddings=2048, attention_dropout=0.0,
        attention_bias=False, use_cache=False,
    )
    config._attn_implementation = "eager"
    reference_module = LlamaAttention(config, layer_idx=0).eval()
    with torch.no_grad():
        for parameter in reference_module.parameters():
            torch.nn.init.normal_(parameter, mean=0.0, std=config.initializer_range)
    state = reference_module.state_dict()
    weight_hash = hashlib.sha256(b"".join(tensor.numpy().tobytes() for tensor in state.values())).hexdigest()
    modules = {}
    for backend in BACKENDS:
        backend_config = copy.deepcopy(config)
        backend_config._attn_implementation = backend
        module = LlamaAttention(backend_config, layer_idx=0).eval()
        module.load_state_dict(state, strict=True)
        modules[backend] = module.to(device="cuda", dtype=torch.float16)
    for module in modules.values():
        assert all(torch.equal(value, modules["eager"].state_dict()[key]) for key, value in module.state_dict().items())

    with torch.inference_mode():
        hidden_states = torch.randn(2, 512, 1024, device="cuda", dtype=torch.float16)
        positions = torch.arange(512, device="cuda").unsqueeze(0).expand(2, -1)
        rotary = LlamaRotaryEmbedding(config=config, device="cuda")
        position_embeddings = rotary(hidden_states, positions)
        # The surrounding model normally prepares these inputs before calling Attention.
        causal_mask = torch.full((512, 512), float("-inf"), device="cuda", dtype=torch.float16)
        causal_mask = torch.triu(causal_mask, diagonal=1)[None, None, :, :]
        arguments = {
            backend: {
                "hidden_states": hidden_states,
                "position_embeddings": position_embeddings,
                "attention_mask": causal_mask if backend == "eager" else None,
                "output_attentions": False,
                "use_cache": False,
            }
            for backend in BACKENDS
        }
        outputs = {backend: module(**arguments[backend])[0] for backend, module in modules.items()}
        reference = outputs["eager"].float()
        equivalence = {}
        for backend, output in outputs.items():
            difference = output.float() - reference
            torch.testing.assert_close(output, outputs["eager"], atol=2e-3, rtol=2e-3)
            equivalence[backend] = {
                "max_abs_error": difference.abs().max().item(),
                "relative_l2": (difference.norm() / reference.norm()).item(),
                "atol": 2e-3, "rtol": 2e-3, "passed": True,
            }
        # Verify causal semantics: changing later tokens cannot change earlier outputs.
        changed_input = hidden_states.clone()
        changed_input[:, 256:, :] += 0.5
        for backend, module in modules.items():
            changed_args = {**arguments[backend], "hidden_states": changed_input}
            changed_output = module(**changed_args)[0]
            torch.testing.assert_close(changed_output[:, :256], outputs[backend][:, :256], atol=2e-3, rtol=2e-3)
            equivalence[backend]["causal_prefix_check"] = True

        result = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "gpu": {**gpu, "sm_count": torch.cuda.get_device_properties(0).multi_processor_count},
            "environment": {
                "prefix": os.sys.prefix, "python": platform.python_version(),
                **{name: importlib.metadata.version(name) for name in ("torch", "transformers", "flash-attn")},
                "torch_cuda": torch.version.cuda,
                "driver": nvidia_query("gpu", "driver_version")[0][0],
            },
            "configuration": {
                "module": "transformers.models.llama.modeling_llama.LlamaAttention",
                "implementation_file": inspect.getfile(LlamaAttention),
                "input_shape": [2, 512, 1024], "heads": 16, "kv_heads": 16,
                "head_dim": 64, "dtype": "float16", "causal": True,
                "dropout": 0.0, "gradients": False, "kv_cache": False,
                "weights": "synthetic normal initialization, identical across backends",
                "weights_fp32_sha256": weight_hash, "seed": seed,
                "trace_forwards_per_backend": 1, "warmup_forwards": 20,
                "precomputed_outside_forward": ["RoPE cos/sin", "eager causal mask"],
            },
            "correctness": equivalence, "backends": {},
        }
        for backend, module in modules.items():
            forward = lambda: module(**arguments[backend])
            for _ in range(20):
                forward()
            torch.cuda.synchronize()
            timing = measure_forward(forward)
            destination = args.out / backend / "trace_view.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            profile_forward(module, arguments[backend], backend, destination)
            result["backends"][backend] = {"timing": timing, **inspect_trace(destination, backend)}
            write_json(args.out / "results.json", result)
            print(f"{backend}: {destination.resolve()}", flush=True)
        write_json(args.out / "results.json", result)


if __name__ == "__main__":
    main()
