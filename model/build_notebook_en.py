#!/usr/bin/env python3
"""Build the academic-English report notebook at the repository root."""
from __future__ import annotations

import nbformat as nbf

OUT = "flash_attn_expmul.ipynb"
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))

md(r"""
# Fused Exponential–Multiplication Operators for FlashAttention-2: A Reproduction Study

**Paper.** K. Alexandridis, V. Titopoulos, and G. Dimitrakopoulos,
*Low-Cost FlashAttention with Fused Exponential and Multiplication Hardware Operators*,
arXiv:2505.14314 [cs.AR], 2025 (IEEE Xplore 11130263).
Reference implementation: <https://github.com/ic-lab-duth/Fused-ExpMul> (MIT).

**Scope.** This notebook reproduces the paper end to end: the `ExpMul` operator (Algorithm 3),
its integration into a FlashAttention-2 kernel (Algorithm 4), language-model accuracy on GLUE,
and RTL synthesis / place-and-route / switching-activity-based power for a 45 nm open-cell
library. All numbers below are produced by the code in this repository and are stored under
`results/`.

**Headline results.**

| Quantity | Baseline | ExpMul | Change |
|---|---|---|---|
| Operator error vs. exact $e^xV$ (RMS) | — | 21.2 % (exponent-only) / 3.3 % (mantissa-interpolated) | — |
| FA-2 kernel output error, $d=64$, $N=512$, FP32 | 0.08 % | 22.5 % / 5.1 % | — |
| FA-2 kernel output error, $d=64$, $N=512$, BF16 | 14.4 % | 12.0 % (mi) | — |
| Lane area, FP32 (post-P&R, 45 nm) | 30 539 µm² | 9 362 µm² | **−69.3 %** |
| Lane power, FP32 (post-P&R, VCD, 50 MHz) | 3.42 mW | 1.02 mW | **−70.2 %** |
| Lane area, BF16 | 9 041 µm² | 5 092 µm² | **−43.7 %** |
| Lane power, BF16 | 0.931 mW | 0.554 mW | **−40.5 %** |
""")

md(r"""
## 1. Method

### 1.1 FlashAttention-2 inner loop (Algorithm 2)

For a query $q$ and key/value vectors $(k_i, v_i)$:

$$
s_i=\langle q,k_i\rangle,\qquad
m_i=\max(m_{i-1},s_i),\qquad
\ell_i=\ell_{i-1}e^{m_{i-1}-m_i}+e^{s_i-m_i},\qquad
o_i=o_{i-1}e^{m_{i-1}-m_i}+v_i e^{s_i-m_i},
$$

with $\mathrm{attn}=o_N/\ell_N$. The computation order is preserved exactly in the C++ model,
the RTL, and the PyTorch attention replacement.

### 1.2 Fusing the scalar exponential with the vector multiplication (Eqs. 2–5)

Merging the running sum and the running output, $\sigma^*_i=[\ell_i,\;o_i]$ and $v^*_i=[1,\;v_i]$:

$$
\sigma^*_i=\sigma^*_{i-1}e^{m_{i-1}-m_i}+v^*_ie^{s_i-m_i}
=\mathrm{ExpMul}(m_{i-1}-m_i,\sigma^*_{i-1})+\mathrm{ExpMul}(s_i-m_i,v^*_i),
\qquad \mathrm{ExpMul}(x,V)=e^xV .
$$

### 1.3 Fixed-point ExpMul (Algorithm 3)

```
x_clip = Clip(x, -15, 0)
x_hat  = Fixed(x_clip)                      # Q6.10, 16-bit
S      = x_hat + (x_hat >> 1) - (x_hat >> 4)  # ≈ -x·log2(e)
L      = round(-S / 2^10)                   # integer number of binary steps
Out    = Float(S_V, E_V - L, M_V)           # exponent-field update only
```

$L$ must be an integer because Eq. (9) applies it as an exponent offset; since $\hat{x}$ is Q6.10,
the product $S$ is Q6.10 as well and the division by $2^{10}$ is implicit. The paper describes
the result as the *rounded* negative value, so round-to-nearest is the default; a `ceil` variant
(the literal reading of $\lfloor\cdot\rfloor$) is kept for ablation.

**A second reading of the operator.** Inspection of the released implementation shows that the
fractional residual is *not* discarded: the integer representation of $V$ is offset by
$-x\log_2 e\cdot 2^{M}$ ($M=23$ for FP32, $7$ for BF16), so the integer part shifts the exponent
and the fractional part interpolates $2^{-f}$ linearly in the mantissa. This variant is still pure
integer shift-and-add and requires no dequantisation, but it is markedly more accurate. Both are
implemented (`EXPMUL_EXPONENT` and `EXPMUL_MANTISSA`) and reported separately, because only the
latter is consistent with the accuracy reported in the paper's Table I.
""")

md(r"""
## 2. Experimental setup

| Item | Value |
|---|---|
| Host / build machine | Windows 11 + MSVC 19.51; Ubuntu 24.04 + g++ 13.3 |
| Simulation | Icarus Verilog 12.0, Verilator 5.020 |
| Synthesis / P&R | OpenROAD-flow-scripts (Yosys 0.68 + ABC, OpenROAD), nangate45 (45 nm, 1.1 V) |
| Library availability | no open 28 nm PDK exists; a 45 nm library is used and only relative savings are claimed |
| RNG seeds | 20250520 (all C++ sweeps), 777 (stability cases), 424242 (per-query detail) |
| Models | `google/flan-t5-base` (248 M parameters), GLUE validation split, 100 examples per task |

Reproduction commands are listed in §9.
""")

md(r"""
## 3. Operator-level accuracy

Two million random pairs $(x,V)$, with $x$ drawn from a mixture of uniform distributions on
$[-15,0]$ and $V$ with random sign, exponent and mantissa. The relative error is measured against
the double-precision value of $e^xV$.

| Variant | mean | RMS | p90 | p99 | max |
|---|---|---|---|---|---|
| Exponent-only (Alg. 3, default) | +0.0000 | **0.2117** | 0.3412 | 0.4298 | 0.4925 |
| Exponent-only with `ceil` | +0.0003 | 0.3091 | 0.4572 | 0.4913 | 0.4998 |
| Mantissa-interpolated (released code) | −0.0000 | **0.0331** | 0.0547 | 0.0855 | 0.1202 |

The exponent-only error is structural: the weight is quantised to an integer power of two, so the
ratio lies in $[2^{-1/2},2^{1/2}]$. Boundary behaviour is exact where it matters: $x=0$ returns
$V$ bit-for-bit, $x=-15$ saturates at $2^{-22}$, positive inputs are clipped to zero (no positive
exponent overflow), subnormal inputs return zero, and no NaN or infinity is ever produced.

![Weight quantisation and relative error](results/phase1/phase1_weight_ratio.png)

![Error distribution over 2 M random pairs](results/phase1/phase1_error_hist.png)

The C++ model is bit-exact against an independent NumPy model over 100 000 vectors for all three
variants (`model/expmul_ref.py --check`).
""")

md(r"""
## 4. FlashAttention-2 kernel

Both versions share one driver (`src/flash_attn_kernel.hpp`): $P$ query vectors are pre-loaded,
key/value vectors are streamed once and broadcast to all $P$ blocks (Fig. 1/2), and only the
update functor differs. The baseline evaluates two piecewise-linear exponentials (8 segments,
maximum error 0.094 %) and performs FP32 multiply-accumulate; the ExpMul version issues two
ExpMul operators and one integer exponent update per lane.

Per-query relative $L_2$ error against a double-precision reference with the same loop order
($d=64$, temperature 1):

| Implementation | $N=16$ | $N=64$ | $N=128$ | $N=512$ |
|---|---|---|---|---|
| Baseline FP32 | 0.05 % | 0.07 % | 0.07 % | **0.08 %** |
| ExpMul, exponent-only | 12.3 % | 19.7 % | 18.6 % | 22.5 % |
| ExpMul, mantissa-interpolated | 3.3 % | 4.5 % | 4.6 % | **5.1 %** |
| Baseline BF16 | 0.69 % | 1.16 % | 1.70 % | 14.4 % |
| ExpMul BF16, mantissa-interpolated | 2.68 % | 4.38 % | 4.49 % | 12.0 % |

![Kernel error versus sequence length](results/phase2/phase2_error_vs_N.png)

Two observations. First, the exponent-only variant is not accurate enough to support the claim of
an accuracy-neutral substitution; the mantissa-interpolated variant is. Second, under BF16 the
quantisation error of the baseline's PWL exponential is random, whereas the logarithmic
quantisation of ExpMul is a smooth bias that partially cancels in $o/\ell$; at $N=512$ the ExpMul
kernel is therefore *more* accurate than the BF16 baseline.

**Numerical stability.** With scores scaled to $10^3$ and $10^6$, all three implementations agree
to about $10^{-5}$ in absolute terms, while a naive softmax without max subtraction overflows in
FP32. No NaN or infinity occurs in any case; the block-parallel configuration ($P=4$) is bit-identical
to the sequential one ($P=1$).

![Stability cases](results/phase2/phase2_stability.png)
""")

md(r"""
## 5. Language-model accuracy on GLUE

Every `T5Attention.forward` call is replaced by a FlashAttention-2 kernel whose softmax weights
are produced by ExpMul (bit-exact with `src/expmul.hpp`); the rest of the model is untouched.
Labels are scored by teacher forcing, so no autoregressive decoding is required. One hundred
validation examples per task are used, selected with a fixed seed.

| Task | FP32 | FP32 exp | FP32 mi | BF16 | BF16 mi |
|---|---|---|---|---|---|
| SST-2 (accuracy) | 81.0 | 79.0 | **82.0** | 81.0 | 81.0 |
| MNLI-m (accuracy) | 37.0 | 36.0 | 37.0 | 37.0 | 37.0 |
| MNLI-mm (accuracy) | 28.0 | 29.0 | 29.0 | 29.0 | 29.0 |

![GLUE accuracy](results/phase3/phase3_glue.png)

**Interpretation.** The absolute levels are not comparable with Table I of the paper: the
reference uses the PromptBench few-shot protocol and a larger Flan-T5, whereas this study uses
zero-shot prompting with the base model (three-way MNLI at 37 % indicates a prompt/scale
mismatch). Differences between variants are at most two points, while the 95 % binomial interval
for $n=100$ is roughly ±8 points; the comparison is therefore indicative, not conclusive. The
direction is consistent with the kernel study: `mi` tracks FP32, `exp` is slightly lower on SST-2.
A ±1 % claim would require `flan-t5-large`/`xl`, few-shot prompts and at least 1000 examples
per task.
""")

md(r"""
## 6. Hardware: synthesis, place-and-route, and power

### 6.1 RTL

| File | Contents |
|---|---|
| `rtl/fp32_funcs.vh` | IEEE-754 binary32 multiply/add as Verilog functions (RNE, flush-to-zero) |
| `rtl/expmul.v` | Algorithm 3, exponent-only (`MODE=0`) and mantissa-interpolated (`MODE=1`) |
| `rtl/exp_pwl.v` | Baseline 8-segment PWL exponential plus a PWL reciprocal |
| `rtl/fa_kernel.v` | One query block: $D$ lanes, pipelined adder tree, single-cycle update |
| `rtl/bf16_funcs.vh`, `rtl/bf16_ops.v`, `rtl/fa_lane_bf16.v` | BF16 datapath equivalents |
| `rtl/fa_lane.v` | One $\sigma^*$ update lane, used for the P&R comparison |

The FP32 multiplier and adder are bit-exact against NumPy over 4 000 random vectors; ExpMul
matches the C++ model on 4 999/5 000 (MODE 0) and 4 996/5 000 (MODE 1) vectors, the remainder
being Q6.10 half-way rounding ties.

### 6.2 Primitive and block area

| FP32 mul | FP32 add | PWL exp | PWL reciprocal | ExpMul (MODE 0) | ExpMul (MODE 1) |
|---|---|---|---|---|---|
| 4 069.5 | 1 519.7 | 4 320.9 | 1 668.4 | **699.6** | 856.0 |

Composing the block from these primitives (dot-product tree + $D$ update lanes + final division):

| $d$ | Baseline (µm²) | ExpMul (µm²) | Saving | Paper (FP32) |
|---|---|---|---|---|
| 16 | 317 870 | 206 395 | **35.1 %** | 23.6 % |
| 32 | 626 949 | 412 640 | **34.2 %** | — |
| 64 | 1 245 107 | 825 132 | **33.7 %** | 29.7 % |
| 256 | 4 954 058 | 3 300 081 | **33.4 %** | 33.1 % |

![Primitive and block area](results/phase4/phase4_units.png)

![Block area versus the paper](results/phase4/phase4_area.png)

### 6.3 Post-route area and power

Both lane variants were taken through synthesis, floorplan, placement, CTS, routing and report
generation. The two designs use distinct top-level modules, which is essential: sharing a
`DESIGN_NAME` makes ORFS reuse the synthesised database, and earlier experiments silently
compared a netlist with itself (18 776 instances on both sides). The verified finger-prints are
18 776 vs. 7 436 instances for FP32 and 6 448 vs. 3 972 for BF16.

Power is measured with real switching activity: a behavioural model of the 45 nm library is
generated from Liberty, the post-route netlist is simulated with 300 random vectors, and the
resulting VCD is annotated in OpenROAD (`read_vcd`) together with the extracted SPEF.

| Data type | Variant | Area (µm²) | Saving | Power @ 50 MHz | Saving |
|---|---|---|---|---|---|
| FP32 | baseline | 30 539 | — | 3.42 mW | — |
| FP32 | ExpMul | 9 362 | **−69.3 %** | 1.02 mW | **−70.2 %** |
| BF16 | baseline | 9 041 | — | 0.931 mW | — |
| BF16 | ExpMul | 5 092 | **−43.7 %** | 0.554 mW | **−40.5 %** |

![Area and power, FP32 and BF16](results/phase4/phase4_power.png)

The FP32 gain is larger because six full-precision multipliers and two PWL exponentials are
removed. For BF16 the multipliers are already small, so replacing them with integer shift-and-add
yields a smaller marginal benefit; the absolute area of the BF16 ExpMul lane, however, is only
one sixth of the FP32 baseline.

A 2 ns constraint cannot be met in this 45 nm library (worst slack ≈ −15 ns), which is why the
P&R runs use a 20 ns period; the paper's 500 MHz target relies on a 28 nm process.
""")

md(r"""
## 7. Layout

![Post-route layouts of the four lane designs](results/visuals/layout_compare.png)

![Stage views of the FP32 baseline](results/visuals/stages_fp32_base.png)

GDS files, ORFS stage images, congestion and IR-drop maps are under `results/visuals/`;
the designs themselves are at `results/visuals/gds/<design>/base/6_final.gds`.
""")

md(r"""
## 8. Discussion

1. **Two readings of Algorithm 3.** The textual formulation changes only the exponent field and
   incurs 15–22 % kernel-level error; the released implementation folds the residual into the
   mantissa and incurs 2–5 %. Only the latter can reproduce the accuracy claimed in the paper.
2. **Area saving is higher than reported (34 % vs. 28.8 %).** The composed block model omits
   fixed overheads (control, registers, multiplexing), which the paper's HLS block includes;
   the paper's small-$d$ result (23.6 %) confirms that these overheads dominate there.
3. **Absolute magnitudes agree.** The paper reports $5.3\times10^5$ µm² for $d=64$ FP32 at 28 nm,
   i.e. about $1.38\times10^6$ µm² after scaling to 45 nm; the model gives $1.245\times10^6$ µm².
4. **Power is measured at lane level, extrapolated at block level.** The lane saving is −70.2 %
   (FP32) and −40.5 % (BF16); a block-level saving of roughly −35 % is an extrapolation from the
   area share of the update path, not a measurement of a full kernel.
5. **Language-model evidence is indicative only.** The 100-example protocol cannot resolve
   differences below eight points; the paper's few-shot, full-length protocol should be used for
   a definitive comparison.

## 9. Reproduction

```bash
# C++ models (Windows/MSVC or Linux/g++)
powershell -File scripts/build.ps1        # or: bash scripts/build.sh
./build/expmul results/phase1
./build/tb results/phase2
python model/expmul_ref.py --check results/phase1/expmul_pairs.csv

# RTL vectors and synthesis (inside the ORFS container)
bash scripts/gen_vcd.sh                   # gate-level VCD for both lane designs
bash scripts/synth_units.sh               # primitive areas
python model/area_model.py                # composed block area

# P&R and power
bash scripts/setup_orfs_bf16.sh && /home/peng/run_orfs.sh ./designs/nangate45/fa_bf16_e/config.mk
bash scripts/power_bf16.sh                # read_vcd + report_power

# GLUE
venv/bin/python t5_glue.py --model models/flan-t5-base --data_dir data/glue \
    --n 100 --variants fp32 fp32_exp fp32_mi bf16 bf16_mi
```

## 10. References

```bibtex
@article{alexandridis2025expmul,
  title={Low-Cost FlashAttention with Fused Exponential and Multiplication Hardware Operators},
  author={Alexandridis, Kosmas and Titopoulos, Vasileios and Dimitrakopoulos, Giorgos},
  journal={arXiv preprint arXiv:2505.14314}, year={2025}}
```

- T. Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*,
  ICLR 2024, arXiv:2307.08691.
- W. Wang et al., *SOLE: Hardware-Software Co-design of Softmax and LayerNorm for Efficient
  Transformer Inference*, ICCAD 2023 (source of the 1.4375 shift-and-add constant).
- C. Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text
  Transformer*, JMLR 21(140), 2020.
- H. W. Chung et al., *Scaling Instruction-Finetuned Language Models*, JMLR 25(70), 2024.
- A. Wang et al., *GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language
  Understanding*, arXiv:1804.07461, 2018.
- K. Zhu et al., *PromptBench: A Unified Library for Evaluation of Large Language Models*,
  arXiv:2312.07910, 2023.
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python", "version": "3.12"}
nbf.write(nb, OUT)
print("wrote", OUT, "cells:", len(cells))
