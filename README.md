# FlashAttention + ExpMul 硬件算子复现

复现论文：

> K. Alexandridis, V. Titopoulos, and G. Dimitrakopoulos,
> “Low-Cost FlashAttention with Fused Exponential and Multiplication Hardware Operators,”
> arXiv:2505.14314 [cs.AR], May 2025.
> IEEE Xplore: <https://ieeexplore.ieee.org/document/11130263>
> 官方开源实现（MIT）：<https://github.com/ic-lab-duth/Fused-ExpMul>

核心思想：把 FlashAttention-2 内层循环里的 `e^x · V`（标量指数 × 向量）融合成一个
**纯整数移位加 + 指数域加法** 的算子 `ExpMul`，输出直接是浮点格式、无需反量化；
从而省掉独立的 exp 单元与浮点乘法器。论文报告 28 nm 下 **面积 −28.8%、功耗 −17.6%**。

---

## 目录结构

```
FlashAttention_accelerator/
├── paper/                    论文 PDF 快照、抽取的正文、Fig.1–4 原图
│   ├── expmul_flashattn_2025.pdf
│   ├── paper_text.txt
│   └── figures/
├── src/                      C++（HLS 风格）参考模型与测试平台
│   ├── expmul.hpp            ExpMul 算子（位精确，FP32/BF16，三种变体）
│   ├── exp_pwl.hpp           baseline 的 8 段分段线性 exp 单元
│   ├── flash_attn_kernel.hpp 两个版本共用的 FA2 driver + 两个 updater
│   ├── flash_attn.hpp        对外接口声明
│   ├── expmul.cpp            阶段 1 误差测试驱动
│   ├── flash_attn_baseline.cpp   独立 exp + FP 乘法
│   ├── flash_attn_expmul.cpp     ExpMul 版 kernel
│   └── tb.cpp                    共享测试平台（阶段 2）
├── model/                    Python 位精确模型 + 绘图
│   ├── expmul_ref.py
│   ├── plot_phase1.py
│   └── plot_phase2.py
├── rtl/                      阶段 2/4：可综合 Verilog（HLS 的 RTL 等价物）
├── scripts/                  构建 / 综合 / 功耗脚本
├── results/                  分阶段结果（表格、CSV、图）
├── notebook/                 汇报 notebook（阶段 5）
└── _ref_official/            论文官方仓库快照（仅作对照，MIT License）
```

## 复现进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| 1 | ExpMul 算子 + 与浮点 `e^x·V` 的误差测试 | ✅ 完成（见 `results/phase1/README.md`） |
| 2 | 集成进 FlashAttention-2 kernel，功能对比 baseline | ✅ 完成（见 `results/phase2/README.md`） |
| 3 | GLUE / Flan-T5 精度验证（FP32/BF16 × baseline/ExpMul） | ⚠️ 脚本就绪，模型下载中 |
| 4 | RTL 综合 + area 对比（nangate45） | ✅ 完成（见 `results/phase4/README.md`） |
| 5 | 汇总报告、差异分析、notebook | ✅ 初版（`notebook/flash_attn_expmul.ipynb`） |

## 环境与依赖

| 项 | 本机（Windows） | 远端 Ubuntu 24.04（192.168.107.130） |
|---|---|---|
| 编译器 | MSVC 19.51 (`cl` 14.51) | g++ 13.3.0 |
| Python | 3.12.4 / numpy 2.5.2 / matplotlib 3.11.1 | 3.12 |
| 仿真 | — | Verilator 5.020、Icarus 12.0 |
| 综合/实现 | — | OpenROAD-flow-scripts（docker 镜像 `openroad/orfs:local`：OpenROAD master + Yosys 0.68） |
| PDK | — | ASAP7（7 nm 预测库）、nangate45、sky130（**开源世界没有 28 nm PDK**，故以相对节省比例对照论文） |

## 快速开始（阶段 1）

```powershell
# Windows / MSVC
powershell -ExecutionPolicy Bypass -File scripts\build.ps1
.\build\expmul.exe results\phase1

# 位精确交叉验证（C++ vs Python）
python model\expmul_ref.py --check results\phase1\expmul_pairs.csv

# 出图
python model\plot_phase1.py
```

Linux 下：`bash scripts/build.sh && ./build/expmul results/phase1`。

## 与论文的对应关系

| 论文内容 | 本仓库实现 |
|---|---|
| Alg.2 FlashAttention-2 内层循环 | `src/flash_attn_expmul.cpp` / `rtl/fa_kernel.v`（阶段 2） |
| Eq.(2)(3) 合并 `l` 与 `o` 更新 | `o*_i = [l_i, o_i]`、`v*_i = [1, v_i]` |
| Eq.(4)(5) `ExpMul(x,V)=e^x V` | `src/expmul.hpp` |
| Eq.(6)(7) `Log2Exp` 移位加 | `log2exp_q()`（Q6.10 + `x + x>>1 - x>>4`） |
| Eq.(8)(9) / Alg.3 指数域结果 | `shift_fp32()` / `shift_bf16()`（无 dequantization） |
| Fig.1/2 块并行结构 | 阶段 2：query 预加载、KV 广播到并行块 |
| Table I GLUE 精度 | 阶段 3 |
| Fig.3/4 area/power | 阶段 4 |

## 引用

```bibtex
@article{alexandridis2025expmul,
  title   = {Low-Cost FlashAttention with Fused Exponential and Multiplication
             Hardware Operators},
  author  = {Alexandridis, Kosmas and Titopoulos, Vasileios and Dimitrakopoulos, Giorgos},
  journal = {arXiv preprint arXiv:2505.14314},
  year    = {2025}
}
```

- T. Dao, “FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning,”
  ICLR 2024, arXiv:2307.08691.
- W. Wang et al., “SOLE: Hardware-Software Co-design of Softmax and LayerNorm for Efficient
  Transformer Inference,” ICCAD 2023.（`Log2Exp` 移位加常数 1.4375 的来源）
- C. Raffel et al., “Exploring the Limits of Transfer Learning with a Unified Text-to-Text
  Transformer,” JMLR 21(140), 2020.
- H. W. Chung et al., “Scaling Instruction-Finetuned Language Models,” JMLR 25(70), 2024.
- A. Wang et al., “GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language
  Understanding,” arXiv:1804.07461, 2018.
- K. Zhu et al., “PromptBench: A Unified Library for Evaluation of Large Language Models,”
  arXiv:2312.07910, 2023.
