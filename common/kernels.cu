#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>

// All launches use PyTorch's current stream; buffers are allocated by the caller.
__device__ __forceinline__ float warp_sum(float v) {
  for (int d=16; d; d>>=1) v += __shfl_down_sync(0xffffffff, v, d);
  return v;
}
__device__ __forceinline__ float block_sum(float v) {
  __shared__ float sums[32];
  int lane=threadIdx.x&31, warp=threadIdx.x>>5;
  v=warp_sum(v);
  if (!lane) sums[warp]=v;
  __syncthreads();
  v=(threadIdx.x < blockDim.x/32) ? sums[lane] : 0.f;
  if (!warp) v=warp_sum(v);
  return v;
}
// MODE 0: original two coalesced scalar loads; 1: four scalar loads;
// MODE 2: one aligned float4 load. GRID adds a bounded grid-stride loop.
template<int MODE, bool GRID, bool REORDER=false>
__global__ void reduce_main(const float* __restrict__ x, float* y, int64_t n) {
  constexpr int V=(MODE==0 ? 2:4);
  float a=0.f,b=0.f,c=0.f,d=0.f;
  int64_t base=int64_t(blockIdx.x)*blockDim.x*V;
  int64_t step=int64_t(gridDim.x)*blockDim.x*V;
  for (; base<n; base+=step) {
    if constexpr (MODE==2) {
      int64_t i=base+int64_t(threadIdx.x)*4;
      if (i+3<n) {
        float4 v=reinterpret_cast<const float4*>(x)[i/4];
        a+=v.x; b+=v.y; c+=v.z; d+=v.w;
      } else {
        if(i<n) a+=x[i]; if(i+1<n)b+=x[i+1];
        if(i+2<n)c+=x[i+2];
      }
    } else {
      int64_t i=base+threadIdx.x;
      if(i<n)a+=x[i]; if(i+blockDim.x<n)b+=x[i+blockDim.x];
      if constexpr (MODE==1) {
        if(i+2*blockDim.x<n)c+=x[i+2*blockDim.x];
        if(i+3*blockDim.x<n)d+=x[i+3*blockDim.x];
      }
    }
    if constexpr (!GRID) break;
  }
  float v=(a+b)+(c+d);
  if constexpr(REORDER) {
    __shared__ float full[1024];
    full[threadIdx.x]=v;
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride>>=1){
      if(threadIdx.x<stride)full[threadIdx.x]+=full[threadIdx.x+stride];
      __syncthreads();
    }
    v=full[0];
  }else v=block_sum(v);
  if(threadIdx.x==0) {
    if constexpr (GRID) y[blockIdx.x]=v;
    else atomicAdd(y,v);
  }
}
__global__ void reduce_finish(const float* x, float* y, int n) {
  float v=0.f;
  for(int i=threadIdx.x;i<n;i+=blockDim.x)v+=x[i];
  v=block_sum(v);
  if(!threadIdx.x)y[0]=v;
}
void check_tensor(torch::Tensor x, torch::ScalarType dtype) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type()==dtype,
              "expected contiguous CUDA tensor of the required dtype");
}
void reduce_out(torch::Tensor x,torch::Tensor y,torch::Tensor scratch,int mode,int threads,int cap) {
  check_tensor(x,torch::kFloat32); check_tensor(y,torch::kFloat32); check_tensor(scratch,torch::kFloat32);
  TORCH_CHECK(x.device()==y.device() && x.device()==scratch.device(),"device mismatch");
  TORCH_CHECK(y.numel()==1 && x.dim()==1,"invalid reduction shape");
  TORCH_CHECK(threads==128 || threads==256 || threads==512 || threads==1024,"invalid threads");
  TORCH_CHECK(mode>=0 && mode<=7 && cap>0,"invalid reduction mode/cap");
  c10::cuda::CUDAGuard guard(x.device());
  auto s=at::cuda::getCurrentCUDAStream();
  int64_t n=x.numel();
  if(!n){C10_CUDA_CHECK(cudaMemsetAsync(y.data_ptr(),0,4,s));return;}
  int v=(mode==0 || mode==5)?2:4;
  int blocks=(n+threads*v-1)/(threads*v);
  bool grid=(mode==3 || mode==4);
  if(grid)blocks=std::min(blocks,cap);
  TORCH_CHECK(scratch.numel()>=blocks || !grid,"scratch too small");
  if(!grid)C10_CUDA_CHECK(cudaMemsetAsync(y.data_ptr(),0,4,s));
  float* dst=grid?scratch.data_ptr<float>():y.data_ptr<float>();
  bool aligned=(reinterpret_cast<uintptr_t>(x.data_ptr())%16)==0;
  if(mode==0) reduce_main<0,false><<<blocks,threads,0,s>>>(x.data_ptr<float>(),dst,n);
  if(mode==1 || (mode==2 && !aligned)) reduce_main<1,false><<<blocks,threads,0,s>>>(x.data_ptr<float>(),dst,n);
  if(mode==2 && aligned) reduce_main<2,false><<<blocks,threads,0,s>>>(x.data_ptr<float>(),dst,n);
  if(mode==3 || (mode==4 && !aligned)) reduce_main<1,true><<<blocks,threads,0,s>>>(x.data_ptr<float>(),dst,n);
  if(mode==4 && aligned) reduce_main<2,true><<<blocks,threads,0,s>>>(x.data_ptr<float>(),dst,n);
  if(mode==5) reduce_main<0,false,true><<<blocks,threads,0,s>>>(x.data_ptr<float>(),dst,n);
  if(mode==6 || (mode==7 && !aligned)) reduce_main<1,false,true><<<blocks,threads,0,s>>>(x.data_ptr<float>(),dst,n);
  if(mode==7 && aligned) reduce_main<2,false,true><<<blocks,threads,0,s>>>(x.data_ptr<float>(),dst,n);
  if(grid)reduce_finish<<<1,256,0,s>>>(dst,y.data_ptr<float>(),blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void mm_naive(const float* a,const float* b,float* c,int M,int N,int K) {
  int r=blockIdx.y*16+threadIdx.y, col=blockIdx.x*16+threadIdx.x;
  if(r<M && col<N){float v=0.f;for(int k=0;k<K;++k)v=fmaf(a[r*K+k],b[k*N+col],v);c[r*N+col]=v;}
}
__global__ void mm_shared(const float* a,const float* b,float* c,int M,int N,int K) {
  __shared__ float as[32][32],bs[32][32];
  int r=blockIdx.y*32+threadIdx.y,col=blockIdx.x*32+threadIdx.x;
  float v=0.f;
  for(int k0=0;k0<K;k0+=32){
    as[threadIdx.y][threadIdx.x]=(r<M && k0+threadIdx.x<K)?a[r*K+k0+threadIdx.x]:0.f;
    bs[threadIdx.y][threadIdx.x]=(k0+threadIdx.y<K && col<N)?b[(k0+threadIdx.y)*N+col]:0.f;
    __syncthreads();
    #pragma unroll
    for(int k=0;k<32;++k)v=fmaf(as[threadIdx.y][k],bs[k][threadIdx.x],v);
    __syncthreads();
  }
  if(r<M && col<N)c[r*N+col]=v;
}

// Register tile, transposed A in shared memory, vectorized global loads/stores.
template<int BM,int BN,int BK,int TM,int TN,bool VECTOR>
__global__ void mm_register(const float* __restrict__ a,const float* __restrict__ b,
                            float* __restrict__ c,int M,int N,int K) {
  __shared__ float as[BK][BM],bs[BK][BN];
  int tid=threadIdx.x,tr=tid/(BN/TN),tc=tid%(BN/TN);
  int r0=blockIdx.y*BM,c0=blockIdx.x*BN;
  float acc[TM][TN]={};
  for(int k0=0;k0<K;k0+=BK){
    if constexpr (VECTOR) {
      for(int q=tid;q<BM*BK/4;q+=blockDim.x){
        int r=q/(BK/4),k=q%(BK/4)*4;
        float4 v=make_float4(0,0,0,0);
        if(r0+r<M && k0+k+3<K && K%4==0)
          v=*reinterpret_cast<const float4*>(a+(r0+r)*K+k0+k);
        else if(r0+r<M){
          if(k0+k<K)v.x=a[(r0+r)*K+k0+k];
          if(k0+k+1<K)v.y=a[(r0+r)*K+k0+k+1];
          if(k0+k+2<K)v.z=a[(r0+r)*K+k0+k+2];
          if(k0+k+3<K)v.w=a[(r0+r)*K+k0+k+3];
        }
        as[k][r]=v.x;as[k+1][r]=v.y;as[k+2][r]=v.z;as[k+3][r]=v.w;
      }
      for(int q=tid;q<BK*BN/4;q+=blockDim.x){
        int k=q/(BN/4),col=q%(BN/4)*4;
        float4 v=make_float4(0,0,0,0);
        if(k0+k<K && c0+col+3<N && N%4==0)
          v=*reinterpret_cast<const float4*>(b+(k0+k)*N+c0+col);
        else if(k0+k<K){
          if(c0+col<N)v.x=b[(k0+k)*N+c0+col];
          if(c0+col+1<N)v.y=b[(k0+k)*N+c0+col+1];
          if(c0+col+2<N)v.z=b[(k0+k)*N+c0+col+2];
          if(c0+col+3<N)v.w=b[(k0+k)*N+c0+col+3];
        }
        *reinterpret_cast<float4*>(&bs[k][col])=v;
      }
    } else {
      for(int q=tid;q<BM*BK;q+=blockDim.x){int r=q/BK,k=q%BK;as[k][r]=(r0+r<M && k0+k<K)?a[(r0+r)*K+k0+k]:0.f;}
      for(int q=tid;q<BK*BN;q+=blockDim.x){int k=q/BN,col=q%BN;bs[k][col]=(k0+k<K && c0+col<N)?b[(k0+k)*N+c0+col]:0.f;}
    }
    __syncthreads();
    #pragma unroll
    for(int k=0;k<BK;++k){
      float ar[TM],br[TN];
      #pragma unroll
      for(int i=0;i<TM;++i)ar[i]=as[k][tr*TM+i];
      #pragma unroll
      for(int j=0;j<TN;++j)br[j]=bs[k][tc*TN+j];
      #pragma unroll
      for(int i=0;i<TM;++i){
        #pragma unroll
        for(int j=0;j<TN;++j)acc[i][j]=fmaf(ar[i],br[j],acc[i][j]);
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for(int i=0;i<TM;++i){
    int r=r0+tr*TM+i,col=c0+tc*TN;
    if constexpr(VECTOR){
      #pragma unroll
      for(int j=0;j<TN;j+=4){
        if(r<M && col+j+3<N && N%4==0)
          *reinterpret_cast<float4*>(c+r*N+col+j)=make_float4(acc[i][j],acc[i][j+1],acc[i][j+2],acc[i][j+3]);
        else {
          #pragma unroll
          for(int v=0;v<4;++v)if(r<M && col+j+v<N)c[r*N+col+j+v]=acc[i][j+v];
        }
      }
    }else{
      #pragma unroll
      for(int j=0;j<TN;++j)if(r<M && col+j<N)c[r*N+col+j]=acc[i][j];
    }
  }
}

// 128x128x32 CTA, eight warps; FP16 inputs and FP32 accumulators/output.
// WMMA requires aligned dimensions. The Python runner pads odd shapes explicitly.
__global__ void mm_wmma(const half* a,const half* b,float* c,int M,int N,int K) {
  using namespace nvcuda;
  __shared__ __align__(32) half as[128][40],bs[32][136];
  int tid=threadIdx.x,warp=tid/32,wr=warp/2,wc=warp%2;
  int r0=blockIdx.y*128,c0=blockIdx.x*128;
  wmma::fragment<wmma::accumulator,16,16,16,float> acc[2][4];
  #pragma unroll
  for(int i=0;i<2;++i){
    #pragma unroll
    for(int j=0;j<4;++j)wmma::fill_fragment(acc[i][j],0.f);
  }
  for(int k0=0;k0<K;k0+=32){
    for(int q=tid;q<128*32/8;q+=256){
      int r=q/4,k=q%4*8;
      int4 v=make_int4(0,0,0,0);
      if(r0+r<M && k0+k<K)v=*reinterpret_cast<const int4*>(a+(r0+r)*K+k0+k);
      *reinterpret_cast<int4*>(&as[r][k])=v;
    }
    for(int q=tid;q<32*128/8;q+=256){
      int k=q/16,col=q%16*8;
      int4 v=make_int4(0,0,0,0);
      if(k0+k<K && c0+col<N)v=*reinterpret_cast<const int4*>(b+(k0+k)*N+c0+col);
      *reinterpret_cast<int4*>(&bs[k][col])=v;
    }
    __syncthreads();
    #pragma unroll
    for(int kk=0;kk<32;kk+=16){
      wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> af[2];
      wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> bf[4];
      #pragma unroll
      for(int i=0;i<2;++i)wmma::load_matrix_sync(af[i],&as[wr*32+i*16][kk],40);
      #pragma unroll
      for(int j=0;j<4;++j)wmma::load_matrix_sync(bf[j],&bs[kk][wc*64+j*16],136);
      #pragma unroll
      for(int i=0;i<2;++i){
        #pragma unroll
        for(int j=0;j<4;++j)wmma::mma_sync(acc[i][j],af[i],bf[j],acc[i][j]);
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for(int i=0;i<2;++i){
    #pragma unroll
    for(int j=0;j<4;++j){
      int r=r0+wr*32+i*16,col=c0+wc*64+j*16;
      if(r<M && col<N)wmma::store_matrix_sync(c+r*N+col,acc[i][j],N,wmma::mem_row_major);
    }
  }
}
void matmul_out(torch::Tensor a,torch::Tensor b,torch::Tensor c,int mode) {
  TORCH_CHECK(mode>=0 && mode<=5,"invalid matmul mode");
  auto dtype=mode==5?torch::kFloat16:torch::kFloat32;
  check_tensor(a,dtype);check_tensor(b,dtype);check_tensor(c,torch::kFloat32);
  TORCH_CHECK(a.dim()==2 && b.dim()==2 && c.dim()==2,"expected matrices");
  TORCH_CHECK(a.size(1)==b.size(0) && c.size(0)==a.size(0) && c.size(1)==b.size(1),"invalid shapes");
  TORCH_CHECK(a.device()==b.device() && a.device()==c.device(),"device mismatch");
  TORCH_CHECK(a.size(0)>0 && b.size(1)>0 && a.size(1)>0,"empty matmul unsupported");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(a.data_ptr())%16==0 && reinterpret_cast<uintptr_t>(b.data_ptr())%16==0 && reinterpret_cast<uintptr_t>(c.data_ptr())%16==0,"matmul needs aligned base pointers");
  c10::cuda::CUDAGuard guard(a.device());auto s=at::cuda::getCurrentCUDAStream();
  int M=a.size(0),K=a.size(1),N=b.size(1);
  if(mode==0)mm_naive<<<dim3((N+15)/16,(M+15)/16),dim3(16,16),0,s>>>(a.data_ptr<float>(),b.data_ptr<float>(),c.data_ptr<float>(),M,N,K);
  if(mode==1)mm_shared<<<dim3((N+31)/32,(M+31)/32),dim3(32,32),0,s>>>(a.data_ptr<float>(),b.data_ptr<float>(),c.data_ptr<float>(),M,N,K);
  if(mode==2)mm_register<64,64,8,4,4,false><<<dim3((N+63)/64,(M+63)/64),256,0,s>>>(a.data_ptr<float>(),b.data_ptr<float>(),c.data_ptr<float>(),M,N,K);
  if(mode==3)mm_register<128,128,8,8,8,false><<<dim3((N+127)/128,(M+127)/128),256,0,s>>>(a.data_ptr<float>(),b.data_ptr<float>(),c.data_ptr<float>(),M,N,K);
  if(mode==4)mm_register<128,128,8,8,8,true><<<dim3((N+127)/128,(M+127)/128),256,0,s>>>(a.data_ptr<float>(),b.data_ptr<float>(),c.data_ptr<float>(),M,N,K);
  if(mode==5){
    TORCH_CHECK(M%16==0 && N%16==0 && K%32==0,"WMMA requires M,N multiples of 16 and K multiple of 32");
    mm_wmma<<<dim3((N+127)/128,(M+127)/128),256,0,s>>>(reinterpret_cast<half*>(a.data_ptr()),reinterpret_cast<half*>(b.data_ptr()),c.data_ptr<float>(),M,N,K);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("reduce_out",&reduce_out);m.def("matmul_out",&matmul_out);}
