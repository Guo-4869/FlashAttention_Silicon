#!/usr/bin/env python3
"""Build the report notebook (nbformat) for the ExpMul FlashAttention reproduction."""
from __future__ import annotations

import nbformat as nbf

OUT = "notebook/flash_attn_expmul.ipynb"

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# 低成本 FlashAttention：融合指数-乘法算子（ExpMul）复现

**论文**：K. Alexandridis, V. Titopoulos, G. Dimitrakopoulos,
*Low-Cost FlashAttention with Fused Exponential and Multiplication Hardware Operators*,
arXiv:2505.14314 [cs.AR], 2025（IEEE Xplore: 11130263）。

**复现内容**：ExpMul 算子 → FlashAttention-2 kernel → Flan-T5/GLUE 精度 → RTL 综合面积。
**硬件目标**：28 nm、500 MHz、II=1、单 KV 向量 8–12 cycle（本复现用 45 nm 开源库替代，只比较相对比例）。

> 本 notebook 的数值全部来自本仓库 `results/` 下的实测文件；命令可直接在 Linux 构建机上重跑。
""")

md(r"""
## 0. 结论速览

| 项目 | 结果 | 状态 |
|---|---|---|
| ExpMul 算子误差（vs 浮点 $e^xV$） | 指数域 RMS **21.2%**；尾数插值 RMS **3.3%** | ✅ 实测 |
| FlashAttention-2 kernel 误差（d=64,N=512,FP32） | baseline **0.08%**；ExpMul 指数域 **22.5%**；尾数插值 **5.1%** | ✅ 实测 |
| 数值稳定性 | 1e3/1e6 大 score 下三种实现误差 ~1e-5，朴素 softmax 溢出 | ✅ 实测 |
| GLUE / Flan-T5（每任务 100 条） | SST-2 81/79/82；MNLI 37/36/37（差异在噪声内） | ✅ 实测（协议受限） |
| 面积（nangate45 组合模型） | 单元级 ExpMul **699.6 µm²** vs exp+乘 **8390 µm²**；块级节省 **33.4–35.1%** | ✅ 实测+组合 |
| 单 lane P&R（FP32 / BF16） | 面积 30 539→9 362 / 9 041→5 092 µm²；功耗 3.42→1.02 / 0.931→0.554 mW | ✅ 实测 |

**核心发现**：论文 Alg.3 写的是“只改指数域”，但其官方实现（github.com/ic-lab-duth/Fused-ExpMul）
把小数残差也加进尾数。前者单元素误差 RMS 21%，在 attention 里放大到 15–22%，
无法解释论文 Table I 的“与 FP32 一致”；后者 RMS 3.3%、kernel 级 2–5%，才是论文方法
真正的等价物。本复现把两者都实现、都评测，并以此为线索解释与论文的差异。
""")

md(r"""
## 1. 论文方法

### 1.1 FlashAttention-2 内层循环（Alg.2）

$$
s_i=\langle q,k_i\rangle,\quad
m_i=\max(m_{i-1},s_i),\quad
l_i=l_{i-1}e^{m_{i-1}-m_i}+e^{s_i-m_i},\quad
o_i=o_{i-1}e^{m_{i-1}-m_i}+v_i e^{s_i-m_i}
$$

最后 $\mathrm{attn}=o_N/l_N$。**计算顺序不可改**，本复现的 C++/RTL 都严格保持该顺序。

### 1.2 合并 l 与 o（Eq.2–3）

令 $\sigma^*_i=[l_i,\ o_i]$、$v^*_i=[1,\ v_i]$：

$$
\sigma^*_i=\sigma^*_{i-1}e^{m_{i-1}-m_i}+v^*_ie^{s_i-m_i}
=\mathrm{ExpMul}(m_{i-1}-m_i,\sigma^*_{i-1})+\mathrm{ExpMul}(s_i-m_i,v^*_i)
$$

### 1.3 ExpMul 的定点实现（Alg.3 / Eq.6–9）

```
x_clip = Clip(x, -15, 0)
x_hat  = Fixed(x_clip)                    # Q6.10, 16-bit
S      = x_hat + (x_hat>>1) - (x_hat>>4)  # ≈ -x·log2(e)
L      = round(-S / 2^10)                 # 整数个 2^-1
Out    = Float(S_V, E_V - L, M_V)         # 只改指数域，直接是浮点，无 dequantization
```

`L` 必须是整数（式 9 把它当指数偏移量），`x̂` 是 Q6.10，所以 `S` 也是 Q6.10，
`−S/2^10` 才得到“几个 2 的幂”。论文正文写的是 “returns its **rounded** negative value”，
即四舍五入；本实现默认四舍五入，另提供 `ceil` 变体作消融。
""")

md(r"""
## 2. 复现环境与命令

| 项 | 值 |
|---|---|
| 本机（Windows） | MSVC 19.51, Python 3.12.4, numpy 2.5.2, matplotlib 3.11.1 |
| 构建机（Ubuntu 24.04） | g++ 13.3, Verilator 5.020, Icarus 12.0, OpenROAD-flow-scripts 镜像（Yosys 0.68 + ABC） |
| 工艺库 | nangate45（45 nm，开源）；28 nm 商业库不可得 |
| 随机种子 | 20250520（所有 C++ 测试）、777（稳定性用例）、424242（逐 query 细节） |

```bash
# 阶段 1/2（Windows 或 Linux）
powershell -File scripts/build.ps1        # 或 bash scripts/build.sh
./build/expmul results/phase1
./build/tb results/phase2
python model/expmul_ref.py --check results/phase1/expmul_pairs.csv   # C++/Python 位精确
# 阶段 4（容器内）
bash scripts/synth_units.sh
python model/area_model.py
```
""")

md(r"""
## 3. 阶段 1：ExpMul 算子误差

2,000,000 组随机 $(x,V)$，$x\sim$ 混合均匀分布 $[-15,0]$，$V$ 随机符号/指数/尾数。

| 变体 | mean | RMS | p90 | p99 | max |
|---|---|---|---|---|---|
| 指数域（Alg.3 字面，主实现） | +0.0000 | **0.2117** | 0.3412 | 0.4298 | 0.4925 |
| 指数域 + ceil 取整（消融） | +0.0003 | 0.3091 | 0.4572 | 0.4913 | 0.4998 |
| 尾数插值（官方仓库写法） | −0.0000 | **0.0331** | 0.0547 | 0.0855 | 0.1202 |

![权重与误差](../results/phase1/phase1_weight_ratio.png)

![误差分布](../results/phase1/phase1_error_hist.png)

边界行为：`x=0` 逐位恒等；`x=-15` 钳位到 $2^{-22}$；正数输入被 clip 成 0（**不产生正指数溢出**）；
次正规输入输出 0；`FLT_MAX` 也不溢出；随机测试 0 个 NaN/Inf。
""")

md(r"""
## 4. 阶段 2：集成进 FlashAttention-2 kernel

两个版本共用同一个 driver（`src/flash_attn_kernel.hpp`），只有 update 算子不同：
baseline 用 8 段 PWL exp + FP32 乘加；ExpMul 用两次 ExpMul + 整数指数更新。
块并行（P 个 query 预加载、K/V 广播）与 Alg.2 顺序完全一致，P=4 与 P=1 结果逐位相同。

指标：每条 query 的 $\|o-\mathrm{ref}\|_2/\|\mathrm{ref}\|_2$，参考为双精度同序 FA2。

| d=64, 温度 1 | N=16 | N=64 | N=128 | N=512 |
|---|---|---|---|---|
| baseline FP32 | 0.05% | 0.07% | 0.07% | **0.08%** |
| ExpMul 指数域 | 12.3% | 19.7% | 18.6% | 22.5% |
| ExpMul 尾数插值 | 3.3% | 4.5% | 4.6% | **5.1%** |
| baseline BF16 | 0.69% | 1.16% | 1.70% | 14.4% |
| ExpMul 尾数插值 BF16 | 2.68% | 4.38% | 4.49% | 12.0% |

![误差随 N](../results/phase2/phase2_error_vs_N.png)

![逐 query 误差](../results/phase2/phase2_error_hist.png)

**稳定性**：score 放大到 $10^3$/$10^6$ 时三种实现最大绝对误差均 ~1e-5，而不减 max 的朴素
softmax 已溢出（`naive_exp_overflow=1`）；全用例无 NaN/Inf。
""")

md(r"""
## 5. 阶段 3：GLUE / Flan-T5（每任务 100 条）

**做法**：把 `T5Attention.forward` 替换为 FA2 + ExpMul 的 torch 实现（其它层不动），
用 teacher-forcing 打分选标签，避免自回归解码。五组对照：FP32、FP32-ExpMul（指数域）、
FP32-ExpMul（尾数插值）、BF16、BF16-ExpMul。模型 `google/flan-t5-base`（248 M），
GLUE 验证集每任务固定 100 条（seed 20250520）。

```python
# 关键：ExpMul 权重（与 src/expmul.hpp 位精确一致）
def expmul_weights(x, mode):
    xc = torch.clamp(x, -15., 0.)
    xh = torch.clamp(torch.floor(xc*1024 + .5), -15360, 0).to(torch.int64)
    p  = -(xh + (xh >> 1) - (xh >> 4))
    if mode == "exp_only":
        return (((127 - ((p + 512) >> 10)) << 23).to(torch.int32).view(torch.float32))
    return ((0x3F800000 - (p << 13)).clamp(min=0).to(torch.int32).view(torch.float32))
```

| 任务 | FP32 | FP32-exp | FP32-mi | BF16 | BF16-mi |
|---|---|---|---|---|---|
| SST-2 (acc) | 81.0 | 79.0 | **82.0** | 81.0 | 81.0 |
| MNLI-m (acc) | 37.0 | 36.0 | 37.0 | 37.0 | 37.0 |
| MNLI-mm (acc) | 28.0 | 29.0 | 29.0 | 29.0 | 29.0 |

![GLUE](../results/phase3/phase3_glue.png)

**必须说明的两点**：

1. 绝对值和论文 Table I（MNLI 87.5）不可比——论文用 PromptBench few-shot + 更大的 Flan-T5；
   这里是 zero-shot base 模型，MNLI 三分类只有 37%（随机 33%），说明 prompt 与模型规模都不匹配。
2. 变体间差异 ≤ 2 个点，而 n=100 的 95% 置信区间约 ±8 点，**统计上不可判定**；
   方向上 `mi ≈ fp32`、`exp` 在 SST-2 上略低，与 kernel 误差（mi 2–5%、exp 15–22%）一致。
   要复现论文的 ±1% 结论需要 large/xl + few-shot + 每任务 ≥1000 条。
""")

md(r"""
## 6. 阶段 4：RTL 综合与面积

工具链：手写 Verilog → Yosys + ABC → nangate45 标准单元。FP32 乘/加与 numpy float32
**逐位一致**（4000 组）；ExpMul 5000 组中 4999/4996 逐位一致（差例为 Q6.10 半格舍入）。

单元面积（45 nm，µm²）：

| FP32 乘 | FP32 加 | PWL exp | rcp | **ExpMul(MODE0)** | ExpMul(MODE1) |
|---|---|---|---|---|---|
| 4069.5 | 1519.7 | 4320.9 | 1668.4 | **699.6** | 856.0 |

块级组合（点积树 + D lane update + 末尾除法）：

| d | baseline | ExpMul | 节省 | 论文 FP32 |
|---|---|---|---|---|
| 16 | 317 870 | 206 395 | **35.1%** | 23.6% |
| 32 | 626 949 | 412 640 | **34.2%** | — |
| 64 | 1 245 107 | 825 132 | **33.7%** | 29.7% |
| 256 | 4 954 058 | 3 300 081 | **33.4%** | 33.1% |

单条 lane 的 update 通路：baseline 9 658.7 µm² → ExpMul 2 918.9 µm²（**−69.8%**）。

![单元面积](../results/phase4/phase4_units.png)

![块级面积](../results/phase4/phase4_area.png)

**功耗 / 面积**（post-route + 门级 VCD 活动率 + SPEF，50 MHz）：

| dtype | 变体 | 面积 (µm²) | 功耗 (mW) |
|---|---|---|---|
| FP32 | baseline | 30 539 | 3.42 |
| FP32 | ExpMul | 9 362（−69.3%） | 1.02（−70.2%） |
| BF16 | baseline | 9 041 | 0.931 |
| BF16 | ExpMul | 5 092（−43.7%） | 0.554（−40.5%） |

![功耗](phase4_power.png)

FP32 收益最大（省掉 6 个乘法器 + 2 个 PWL exp）；BF16 因为乘法器本身便宜，边际收益降到 44%。
四个设计都跑到了 `6_final.gds`。块级按 update 通路占比折算约 −35%（高于论文 17.6%，
因为论文块内的控制/寄存器/共享 exp 等固定开销占比更大）。
""")

md(r"""
## 7. 与论文的差异分析

1. **Alg.3 的两种读法**（最重要）。论文正文只改指数域，官方代码还加尾数插值。
   前者 kernel 级误差 15–22%，后者 2–5%；只有后者能解释 Table I 的“与 FP32 基本一致”。
2. **面积比例偏高（34% vs 28.8%）**。我们的组合模型没有计入块内的控制/寄存器/多路选择
   等固定开销；论文 d=16 只节省 23.6%，正说明其固定开销占比更高。
3. **绝对面积量级吻合**。论文 d=64 FP32 基线 $5.3\times10^5$ µm²@28nm，折算到 45nm 约
   $1.38\times10^6$，我们组合出 $1.245\times10^6$（差 10%）。
4. **功耗已实测（单 lane）**：FP32 −70.2%、BF16 −40.5%（门级 VCD + SPEF，post-route）。
5. **BF16 已补齐**：`rtl/bf16_funcs.vh` + `bf16_ops.v` + `fa_lane_bf16.v`，
   面积 9 041 → 5 092 µm²（−43.7%），因为 bf16 乘法器本身便宜，边际收益小于 FP32。

## 8. 版图与可视化

四个 lane 设计都跑完 ORFS 全流程（含 GDS），图在 `results/visuals/`：

![版图对比](../results/visuals/layout_compare.png)

FP32 baseline（30.5k µm²）的布局最满，ExpMul 版（9.4k µm²）明显稀疏，BF16 两版更紧凑。
同一设计的阶段视图（placement / congestion / IR-drop / routing）：

![阶段视图](../results/visuals/stages_fp32_base.png)

GDS 已拉回本地：`results/visuals/gds/<design>/base/6_final.gds`（KLayout 可直接打开），
原始 40 张 ORFS 报告图在 `results/visuals/<design>/base/`。

## 9. 引用

```bibtex
@article{alexandridis2025expmul,
  title={Low-Cost FlashAttention with Fused Exponential and Multiplication Hardware Operators},
  author={Alexandridis, Kosmas and Titopoulos, Vasileios and Dimitrakopoulos, Giorgos},
  journal={arXiv preprint arXiv:2505.14314}, year={2025}}
```
- T. Dao, *FlashAttention-2*, ICLR 2024, arXiv:2307.08691.
- W. Wang et al., *SOLE: Hardware-Software Co-design of Softmax and LayerNorm*, ICCAD 2023
  （`Log2Exp` 常数 1.4375 的来源）。
- C. Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text
  Transformer*, JMLR 21(140), 2020.
- H. W. Chung et al., *Scaling Instruction-Finetuned Language Models*, JMLR 25(70), 2024.
- A. Wang et al., *GLUE*, arXiv:1804.07461, 2018.
- K. Zhu et al., *PromptBench*, arXiv:2312.07910, 2023.
- 官方实现：https://github.com/ic-lab-duth/Fused-ExpMul （MIT）
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python", "version": "3.12"}
nbf.write(nb, OUT)
print("wrote", OUT, "cells:", len(cells))
