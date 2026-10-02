"""Shared reproducible timing and metadata. No profiler inside timed regions."""
import datetime, hashlib, json, os, platform, random, statistics, subprocess
from pathlib import Path
import torch
from torch.utils.cpp_extension import load

ROOT=Path(__file__).resolve().parents[1]

def setup(seed=20260929):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False

def extension():
    os.environ.setdefault('MAX_JOBS','4')
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST','8.0')
    return load(name='fudan_homework_cuda_v1',sources=[str(ROOT/'common/kernels.cu')],
                extra_cuda_cflags=['-O3','-lineinfo','--ptxas-options=-v'],verbose=True)

def metadata():
    p=torch.cuda.get_device_properties(0)
    def shell(cmd):
        r=subprocess.run(cmd,shell=True,capture_output=True,text=True)
        return (r.stdout+r.stderr).strip()
    return {'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'host':platform.node(),'python':platform.python_version(),'torch':torch.__version__,
        'torch_cuda':torch.version.cuda,'device':p.name,'sm_count':p.multi_processor_count,
        'compute_capability':f'{p.major}.{p.minor}','vram_bytes':p.total_memory,
        'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'nvcc':shell('/usr/local/cuda/bin/nvcc --version'),
        'nvidia_smi':shell('nvidia-smi'),'seed':20260929,'tf32':False,
        'fp16_reduced_precision_reduction':False,
        'kernels_sha256':hashlib.sha256((ROOT/'common/kernels.cu').read_bytes()).hexdigest()}

def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False))

def time_cuda(fn, *, repeats=9, target_ms=30, max_inner=200, graph=True):
    """Median of independent event batches; graph replay amortizes Python launch gaps.

    The same input/output buffers are reused (warm-cache). All samples are retained.
    Includes all GPU operations in fn, e.g. memset and second reduction pass.
    """
    for _ in range(10): fn()
    torch.cuda.synchronize()
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5): fn()
    end.record();end.synchronize()
    estimate=max(start.elapsed_time(end)/5,0.002)
    inner=min(max_inner,max(1,int(target_ms/estimate)))
    if graph:
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(inner):fn()
        run=g.replay
        for _ in range(3):run()
        torch.cuda.synchronize()
    else:
        def run():
            for _ in range(inner):fn()
    samples=[]
    for _ in range(repeats):
        start.record();run();end.record();end.synchronize()
        samples.append(start.elapsed_time(end)*1000/inner)
    med=statistics.median(samples)
    return {'median_us':med,'min_us':min(samples),'max_us':max(samples),
            'p25_us':sorted(samples)[len(samples)//4],
            'p75_us':sorted(samples)[3*len(samples)//4],
            'samples_us':samples,'inner':inner,'repeats':repeats,
            'method':'CUDA events / CUDA Graph batch' if graph else 'CUDA events / eager batch'}
