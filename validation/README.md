# 验证证据

- `memcheck.log`：实际运行 `compute-sanitizer --tool memcheck --error-exitcode 99`，归约和矩阵乘各出现一次 `ERROR SUMMARY: 0 errors`。
- `reduction_memcheck/correctness.json`：342 个数值用例；`matmul_memcheck/correctness.json`：36 个数值用例。
- `kernels.sass.txt`：通过 nvcc 编译后的扩展执行 `cuobjdump --dump-sass` 得到，包含 vectorized load/store 以及 WMMA 的 HMMA 指令。
- `ncu_reduction.csv` / `ncu_reduction.log`：Nsight Compute 的原始运行输出；因为 `ERR_NVGPUCTRPERM` 未产生硬件计数器测量。CSV 文件保留原名，但其内容为错误消息，不能作为指标数据读取。
- `artifact_checks.json`：本地交付一致性检查，包含 PDF 页数、字体、文本边界、数值记录数、哈希及源码匹配。
- `visual_review.md`：报告渲染后的人工视觉检查记录。

主实验使用物理 GPU 2；独立 memcheck 使用物理 GPU 4；受限 ncu 尝试使用物理 GPU 3。没有使用同一 GPU 同时运行性能基准和 sanitizer。

## 管理员开放计数器后可选重跑

```bash
source /225045001/miniconda3/bin/activate shenjiarun
cd /225045001/fudan-nlp-ai-infra-homework-20260929
export CUDA_VISIBLE_DEVICES=3 CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=4
ncu --target-processes all --kernel-name regex:reduce_main \
  --launch-count 1 --section LaunchStats --section Occupancy \
  --section SpeedOfLight --csv \
  python homework2/run.py --ncu > validation/ncu_reduction_rerun.csv
ncu --target-processes all --kernel-name regex:mm_register \
  --launch-count 1 --section LaunchStats --section Occupancy \
  --section SpeedOfLight --csv \
  python homework3/run.py --ncu > validation/ncu_matmul_rerun.csv
```

执行前确认 GPU 3 仍空闲。这些计数器不是当前 PDF 中数值的来源，不能将重跑数据悄悄替换为原始测量。
