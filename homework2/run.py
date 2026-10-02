"""Homework 2: sum reduction, scalar/float4 ablations and correctness checks."""

import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from common.bench import setup, extension, metadata, dump, time_cuda

VARIANTS = {
    "scalar2_atomic": (0, 256),
    "scalar4_atomic": (1, 256),
    "float4_atomic": (2, 256),
    "scalar4_two_pass": (3, 256),
    "float4_two_pass": (4, 256),
    "scalar2_atomic_1024": (0, 1024),
    "reorder_scalar2": (5, 1024),
    "reorder_scalar4": (6, 1024),
    "reorder_float4": (7, 1024),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="homework2/results/a800")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--ncu", action="store_true")
    args = ap.parse_args()
    setup()
    mod = extension()
    meta = metadata()
    cap = meta["sm_count"] * 8
    scratch = torch.empty(cap, device="cuda")
    y = torch.empty((), device="cuda")
    if args.ncu:
        x = torch.randn(1 << 26, device="cuda")
        mod.reduce_out(x, y, scratch, 4, 256, cap)
        torch.cuda.synchronize()
        return
    checks = []
    for n in [
        0,
        1,
        2,
        3,
        4,
        31,
        32,
        33,
        255,
        256,
        257,
        1023,
        1024,
        1025,
        4099,
        65537,
        1048579,
    ]:
        for offset in [0, 1]:
            storage = torch.randn(n + offset, device="cuda")
            x = storage[offset:]
            ref = x.double().sum().item()
            scale = x.double().abs().sum().item()
            for name, (mode, threads) in VARIANTS.items():
                mod.reduce_out(x, y, scratch, mode, threads, cap)
                got = y.item()
                err = abs(got - ref)
                bound = 2e-6 * scale + 1e-6
                assert err <= bound, (n, offset, name, got, ref, bound)
                checks.append(
                    {
                        "N": n,
                        "offset": offset,
                        "variant": name,
                        "ref_fp64": ref,
                        "result": got,
                        "abs_error": err,
                        "sum_abs": scale,
                        "bound": bound,
                        "pass": True,
                    }
                )
    for kind in ["ones", "alternating", "cancellation", "large_magnitude"]:
        n = 1048579
        x = torch.ones(n, device="cuda")
        if kind == "alternating":
            x[1::2] = -1
        if kind == "cancellation":
            x[::4] = 1e6
            x[1::4] = 1
            x[2::4] = -1e6
            x[3::4] = 1
        if kind == "large_magnitude":
            x = torch.randn(n, device="cuda") * 1e10
        ref = x.double().sum().item()
        scale = x.double().abs().sum().item()
        for name, (mode, threads) in VARIANTS.items():
            mod.reduce_out(x, y, scratch, mode, threads, cap)
            got = y.item()
            err = abs(got - ref)
            assert err <= 2e-6 * scale + 1e-6, (kind, name, got, ref)
            checks.append(
                {
                    "kind": kind,
                    "N": n,
                    "variant": name,
                    "ref_fp64": ref,
                    "result": got,
                    "abs_error": err,
                    "sum_abs": scale,
                    "bound": 2e-6 * scale + 1e-6,
                    "pass": True,
                }
            )
    rejected = []
    for kind, x in [
        ("noncontiguous", torch.randn(20, device="cuda")[::2]),
        ("float64", torch.randn(20, device="cuda", dtype=torch.float64)),
    ]:
        try:
            mod.reduce_out(x, y, scratch, 4, 256, cap)
        except RuntimeError:
            rejected.append(kind)
        else:
            raise AssertionError("invalid input accepted: " + kind)
    # Verify current-stream correctness, not just default-stream behavior.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.arange(4099, device="cuda", dtype=torch.float32)
        mod.reduce_out(x, y, scratch, 4, 256, cap)
    stream.synchronize()
    assert y.item() == 4098 * 4099 / 2
    out = Path(args.out)
    dump(
        out / "correctness.json",
        {"checks": checks, "rejected": rejected, "nondefault_stream": True},
    )
    print(
        f"Correctness: {len(checks)} numeric checks + rejection/stream checks PASS",
        flush=True,
    )
    if args.check_only:
        return
    rows = []
    sizes = [256, 4096, 65536, 1048576, 16777216, 67108864, 1048579]
    for n in sizes:
        x = torch.randn(n, device="cuda")
        ref = x.double().sum().item()
        scale = x.double().abs().sum().item()
        for name, (mode, threads) in VARIANTS.items():
            fn = lambda: mod.reduce_out(x, y, scratch, mode, threads, cap)
            timing = time_cuda(fn)
            got = y.item()
            assert abs(got - ref) <= 2e-6 * scale + 1e-6
            row = {
                "N": n,
                "variant": name,
                "threads": threads,
                "cap": cap,
                "grid": min((n + threads * 4 - 1) // (threads * 4), cap)
                if mode in (3, 4)
                else (n + threads * (2 if mode in (0, 5) else 4) - 1)
                // (threads * (2 if mode in (0, 5) else 4)),
                "abs_error": abs(got - ref),
                "ref_fp64": ref,
                "result": got,
                "sum_abs": scale,
                "effective_GBps": 4 * n / (timing["median_us"] * 1000),
                **timing,
            }
            if n in [1048576, 67108864]:
                row["eager"] = time_cuda(fn, graph=False, repeats=5, max_inner=100)
            rows.append(row)
            print(n, name, round(timing["median_us"], 3), "us", flush=True)
        fn = lambda: torch.sum(x, dim=(0,), out=y)
        timing = time_cuda(fn)
        rows.append(
            {
                "N": n,
                "variant": "torch_sum",
                "effective_GBps": 4 * n / (timing["median_us"] * 1000),
                "abs_error": abs(y.item() - ref),
                "ref_fp64": ref,
                "result": y.item(),
                **timing,
            }
        )
        dump(
            out / "results.json",
            {
                "metadata": meta,
                "rows": rows,
                "timing_note": "warm-cache graph replay, output reset and second pass included; input/output allocation excluded",
            },
        )
    # Tuning results are separate from fixed-configuration ablations.
    tuning = []
    x = torch.randn(67108864, device="cuda")
    for threads in [128, 256, 512]:
        for blocks_per_sm in [2, 4, 8, 16]:
            grid = meta["sm_count"] * blocks_per_sm
            tmp = torch.empty(grid, device="cuda")
            timing = time_cuda(
                lambda: mod.reduce_out(x, y, tmp, 4, threads, grid), repeats=7
            )
            tuning.append(
                {"threads": threads, "blocks_per_sm": blocks_per_sm, **timing}
            )
    dump(out / "tuning.json", {"N": x.numel(), "rows": tuning})
    from torch.profiler import profile, ProfilerActivity, record_function

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
    ) as p:
        for name, (mode, t) in VARIANTS.items():
            with record_function(name):
                mod.reduce_out(x, y, scratch, mode, t, cap)
        torch.cuda.synchronize()
    p.export_chrome_trace(str(out / "trace.json"))


if __name__ == "__main__":
    main()
