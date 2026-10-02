# 交付前检查

2026-09-29 完成。

## PDF

- 三个 PDF 共 18 页（5 / 6 / 7）。使用 PyMuPDF 将全部页面渲染为 PNG，并检查每页缩略图与关键正文页。
- 中文字体已嵌入；标题、正文、表格、图例、页码完整，未发现重叠、越界、缺字或意外空白页。
- 文本边界自动检查全部通过，详见 `artifact_checks.json`。
- 性能表、图和结论均由保存的 results.json 生成。CUDA 源码 SHA256 与作业二、三实测元数据一致。
- SASS 确认 float4 路径含 `LDG.E.128`，FP32 向量化 GEMM 含 `STG.E.128`，WMMA 含 `HMMA`。

## 文字与表格颜色修改后的复查

- 先恢复首次报告的标题、字号、编号、页脚和 Figure 样式，再仅修改文档文字与数值表格的颜色。
- 正文、标题、链接和表格文字为黑色，数值表格使用白底黑线。
- 八张 Figure 恢复最初的配色、实线与圆形标记、网格和图例样式；Figure 内部的文字与坐标轴配色同样保留原样。
- 三份 PDF 页数仍为 5 / 6 / 7；全部 18 页重新渲染并目视检查。
- 原始数据、实验代码、测量结果和报告内容不变。

## 离线 trace 查看器

- 在 Codex 内置浏览器加载本地 `homework1/trace_viewer.html`，检查了时间线、指标和表格的显示。
- 显式 Attention 首轮显示 13 行；切换 SDPA 显示 3 行，全部采样轮次显示 9 行。
- 点击 SDPA 第二个 kernel，显示 `flash_fwd_kernel`、32.736 μs、grid=[4,2,16]、block=[128,1,1]。
- CSV 下载得到 14 行（1 行表头 + 显式 Attention 的 13 个 kernel），已用 CSV parser 检查列名。
- 检查时浏览器没有 JavaScript error。测试用本地 HTTP server 已结束，不影响直接以文件方式打开 HTML。

## GPU 验证

- 数值测试、接口拒绝、非默认 stream 测试均完成。
- 两项 Compute Sanitizer memcheck 均报告 0 errors。
- Nsight Compute 硬件计数器被权限拒绝，已如实披露，未编造指标。
- 本次实验进程已结束；其他项目此后可重新使用 GPU。复现前仍需重新查看当前空闲卡。
