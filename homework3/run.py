"""Homework 3: generated CUDA GEMM kernels, fair precision-separated baselines."""
import argparse, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from common.bench import setup, extension, metadata, dump, time_cuda

VARIANTS={'naive':0,'shared32':1,'register64':2,'register128':3,'float4_register128':4}

def errors(got,ref):
    delta=(got.double()-ref.double()).abs()
    return {'max_abs_error':delta.max().item(),'rms_error':delta.square().mean().sqrt().item(),
            'relative_l2':(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(ref.double()).clamp_min(1e-30)).item()}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='homework3/results/a800')
    ap.add_argument('--check-only',action='store_true');ap.add_argument('--ncu',action='store_true')
    args=ap.parse_args();setup();mod=extension();meta=metadata();out=Path(args.out)
    if args.ncu:
        a=torch.randn(2048,2048,device='cuda');b=torch.randn_like(a);c=torch.empty_like(a)
        mod.matmul_out(a,b,c,4);torch.cuda.synchronize();return
    checks=[]
    for M,N,K in [(1,1,1),(17,19,13),(65,127,33),(129,131,137),(128,128,128),(256,192,96)]:
        a=torch.randn(M,K,device='cuda');b=torch.randn(K,N,device='cuda');c=torch.empty(M,N,device='cuda')
        ref=a.double()@b.double()
        for name,mode in VARIANTS.items():
            mod.matmul_out(a,b,c,mode);e=errors(c,ref)
            assert torch.allclose(c.double(),ref,atol=2e-4,rtol=2e-4),(M,N,K,name,e)
            checks.append({'shape_MNK':[M,N,K],'variant':name,**e,'pass':True})
        # Padded wrapper: tails zero-filled outside kernel; compare same quantized input.
        ah=a.half();bh=b.half();mp=(M+15)//16*16;np=(N+15)//16*16;kp=(K+31)//32*32
        apad=F.pad(ah,(0,kp-K,0,mp-M));bpad=F.pad(bh,(0,np-N,0,kp-K));cpad=torch.empty(mp,np,device='cuda')
        mod.matmul_out(apad,bpad,cpad,5);refh=ah.double()@bh.double();e=errors(cpad[:M,:N],refh)
        assert torch.allclose(cpad[:M,:N].double(),refh,atol=3e-4,rtol=3e-4),e
        checks.append({'shape_MNK':[M,N,K],'variant':'wmma_padded',**e,'pass':True})
    rejected=[]
    for label,a,b,c,mode in [
        ('mismatched_shapes',torch.empty(3,4,device='cuda'),torch.empty(5,6,device='cuda'),torch.empty(3,6,device='cuda'),4),
        ('noncontiguous',torch.empty(4,4,device='cuda').t(),torch.empty(4,4,device='cuda'),torch.empty(4,4,device='cuda'),4),
        ('wmma_unaligned_dims',torch.empty(17,32,device='cuda',dtype=torch.float16),torch.empty(32,16,device='cuda',dtype=torch.float16),torch.empty(17,16,device='cuda'),5)]:
        try:mod.matmul_out(a,b,c,mode)
        except RuntimeError:rejected.append(label)
        else:raise AssertionError(label+' unexpectedly accepted')
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        a=torch.eye(128,device='cuda');b=torch.randn_like(a);c=torch.empty_like(a);mod.matmul_out(a,b,c,4)
    stream.synchronize();assert torch.equal(b,c)
    dump(out/'correctness.json',{'checks':checks,'rejected':rejected,'nondefault_stream':True})
    print(f'Correctness: {len(checks)} numeric checks + rejection/stream checks PASS',flush=True)
    if args.check_only:return
    rows=[]
    shapes=[(128,128,128),(512,512,512),(1024,1024,1024),(2048,2048,2048),(4096,4096,4096),
            (1024,4096,1024),(1024,1024,4096),(513,769,257)]
    for M,N,K in shapes:
        a=torch.randn(M,K,device='cuda');b=torch.randn(K,N,device='cuda');c=torch.empty(M,N,device='cuda')
        # Full FP64 reference for small cases, strict FP32 cuBLAS plus sampled FP64 for large ones.
        ref=a@b
        ri=torch.linspace(0,M-1,min(32,M),device='cuda').long();ci=torch.linspace(0,N-1,min(32,N),device='cuda').long()
        ref64=a[ri].double()@b[:,ci].double()
        variants=list(VARIANTS.items())+[('torch_fp32',None)]
        for name,mode in variants:
            fn=(lambda:torch.mm(a,b,out=c)) if mode is None else (lambda:mod.matmul_out(a,b,c,mode))
            fn();e=errors(c,ref);sample=errors(c[ri][:,ci],ref64)
            assert e['relative_l2']<2e-5 and sample['relative_l2']<2e-5,(M,N,K,name,e,sample)
            timing=time_cuda(fn,repeats=9,max_inner=100)
            row={'shape_MNK':[M,N,K],'variant':name,'input_dtype':'float32','output_dtype':'float32',
                 'TFLOPS':2*M*N*K/(timing['median_us']*1e6),**e,'sampled_fp64':sample,**timing}
            if M==1024 and N==1024 and K==1024:row['eager']=time_cuda(fn,graph=False,repeats=5,max_inner=100)
            rows.append(row);print([M,N,K],name,round(timing['median_us'],3),'us',flush=True)
        ah=a.half();bh=b.half();refh=ah.float()@bh.float()
        mp=(M+15)//16*16;np=(N+15)//16*16;kp=(K+31)//32*32
        apad=F.pad(ah,(0,kp-K,0,mp-M));bpad=F.pad(bh,(0,np-N,0,kp-K));cpad=torch.empty(mp,np,device='cuda')
        fn=lambda:mod.matmul_out(apad,bpad,cpad,5)
        fn();e=errors(cpad[:M,:N],refh)
        assert e['relative_l2']<2e-5,([M,N,K],e)
        timing=time_cuda(fn,repeats=9,max_inner=100)
        rows.append({'shape_MNK':[M,N,K],'padded_MNK':[mp,np,kp],'variant':'wmma_fp16_fp32out',
                     'input_dtype':'float16','output_dtype':'float32','TFLOPS':2*M*N*K/(timing['median_us']*1e6),**e,**timing})
        # torch.mm output is FP16; report explicitly, not an identical-output baseline.
        ch=torch.empty(M,N,device='cuda',dtype=torch.float16)
        timing=time_cuda(lambda:torch.mm(ah,bh,out=ch),repeats=9,max_inner=100)
        rows.append({'shape_MNK':[M,N,K],'variant':'torch_fp16_fp16out','input_dtype':'float16','output_dtype':'float16',
                     'TFLOPS':2*M*N*K/(timing['median_us']*1e6),**errors(ch,refh),**timing})
        # Matching input/output precision via PyTorch cuBLAS out_dtype (if supported).
        try:
            fn=lambda:torch.mm(ah,bh,out_dtype=torch.float32,out=c)
            fn();timing=time_cuda(fn,repeats=9,max_inner=100)
            rows.append({'shape_MNK':[M,N,K],'variant':'torch_fp16_fp32out','input_dtype':'float16','output_dtype':'float32',
                         'TFLOPS':2*M*N*K/(timing['median_us']*1e6),**errors(c,refh),**timing})
        except (TypeError,RuntimeError) as exc:
            meta['torch_fp32_output_limitation']=str(exc)
        if [mp,np,kp]!=[M,N,K]:
            def padded_e2e():
                aa=F.pad(ah,(0,kp-K,0,mp-M));bb=F.pad(bh,(0,np-N,0,kp-K));cc=torch.empty(mp,np,device='cuda')
                mod.matmul_out(aa,bb,cc,5);return cc[:M,:N].contiguous()
            rows.append({'shape_MNK':[M,N,K],'variant':'wmma_padding_included',**time_cuda(padded_e2e,repeats=9,max_inner=100)})
        dump(out/'results.json',{'metadata':meta,'rows':rows,'timing_note':'warm-cache graph replay; preallocated outputs; no compile/allocation/input conversion in aligned-kernel timings'})
    from torch.profiler import profile,ProfilerActivity,record_function
    a=torch.randn(1024,1024,device='cuda');b=torch.randn_like(a);c=torch.empty_like(a)
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True) as p:
        for name,mode in VARIANTS.items():
            with record_function(name):mod.matmul_out(a,b,c,mode)
        torch.cuda.synchronize()
    p.export_chrome_trace(str(out/'trace.json'))

if __name__=='__main__':main()
