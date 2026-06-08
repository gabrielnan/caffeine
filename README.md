# caffeine: attention optimization problem

**Motivation:** most gradient optimizers are agnostic to values being optimized.
Muon has demonstrated that matrix-specific optimizer can outperform a fully agnostic one.
The goal of this task is to see whether we can create an even better optimizer by finding the best optimizer specifically for attention models.

Task: design an optimizer that trains a fixed vanilla self-attention model on a toy task, each track defines a unique task.

## tracks


| Track | Objective | Target |
| --- | --- | --- |
| `random_teacher` | Train a model to match the outputs of another randomly initialized hidden model (i.e. teacher). | eval MSE <= `4e-7` |
| `single_ar` | Opaque single-query associative recall: `8` pair tokens + `1` query token -> value class. | eval accuracy >= `0.99` |
| `mqar` | Opaque multi-query associative recall: `8` pair tokens + `8` query tokens -> value classes. | eval accuracy >= `0.99` |

`mqar` is inspired by the multi-query associative recall task from
[Zoology: Measuring and Improving Recall in Efficient Language Models](https://arxiv.org/abs/2312.04927).


The benchmark supports multiple tracks through `run_eval.py --track`:

```bash
uv run python run_eval.py --track single_ar --submission submissions/adamw/submission.py
uv run python run_eval.py --track mqar --submission submissions/adamw/submission.py
```


## scoring

Official v0 scoring is wall-clock time on GitHub Actions `macos-15` arm64. The harness uses MPS when available and falls back to CPU. It stops at the first fixed evaluation checkpoint where the selected track's target metric is met; submissions that miss the target within that track's step budget fail.

Run locally:

```bash
uv sync
uv run python test_benchmark.py
uv run python run_eval.py --submission submissions/adamw/submission.py --results-json result.json
```

## synthetic-track solution

```bash
uv run python run_eval.py --track single_ar --submission submissions/qkv_wcos_etf_adam/submission.py
uv run python run_eval.py --track mqar --submission submissions/qkv_wcos_etf_adam/submission.py
```

`qkv_wcos_etf_adam` uses AdamW with ETF readout initialization, fixed row-norm
projection for Q/K/V/readout matrices, and a zero readout bias on the synthetic
token-attention architecture. It falls back to the default AdamW settings on
non-token tracks.

## synthetic-track leaderboards

The ranking metric is `steps`: the first training step where the track target is
met. Token tracks evaluate every step. Lower is better. Failed runs are shown at
`max_steps` and ranked after passing runs.

### `single_ar`

| Rank | Submission | Status | Steps | Best eval accuracy | Notes |
| --- | --- | --- | ---: | ---: | --- |
| 1 | `qkv_wcos_etf_adam` | pass | 228 | 0.990234375 | Main ETF row-projected AdamW submission. |
| 1 | `qkv_comp_muon_residual` | pass | 228 | 0.990234375 | Compositional-Muon-style residual on Q/K and V/readout. |
| 3 | `qkv_wcos_etf_block_beta` | pass | 229 | 0.990234375 | Separate beta2 values by block. |
| 4 | `qkv_wcos_etf_block_lr` | pass | 230 | 0.990234375 | Late Q/K/V/readout LR multipliers. |
| 5 | `qkv_wcos_etf_hadamard` | pass | 239 | 0.990722656 | Hadamard initialization ablation. |
| 6 | `qkv_wcos_etf_adam_s6` | pass | 246 | 0.990478516 | Lower LR, larger readout norm. |
| 6 | `qkv_wcos_etf_muon_v` | pass | 246 | 0.990478516 | Small V Muon residual ablation. |
| 8 | `qkv_row_scalar_wcos_etf` | pass | 250 | 0.990234375 | Row-scalar Q/K/V second moments. |
| - | `adamw_fast_synth` | fail | 400 | 0.973876953 | Plain faster AdamW baseline. |
| - | `adamw` | fail | 400 | 0.619384766 | Default AdamW baseline. |

### `mqar`

| Rank | Submission | Status | Steps | Best eval accuracy | Notes |
| --- | --- | --- | ---: | ---: | --- |
| 1 | `qkv_wcos_etf_adam` | pass | 76 | 0.991546631 | Main ETF row-projected AdamW submission. |
| 1 | `qkv_comp_muon_residual` | pass | 76 | 0.991546631 | Compositional-Muon-style residual on Q/K and V/readout. |
| 3 | `qkv_wcos_etf_hadamard` | pass | 78 | 0.990081787 | Hadamard initialization ablation. |
| 4 | `qkv_row_scalar_wcos_etf` | pass | 79 | 0.990509033 | Row-scalar Q/K/V second moments. |
| 5 | `qkv_wcos_etf_block_lr` | pass | 85 | 0.990081787 | Late Q/K/V/readout LR multipliers. |
| 6 | `qkv_wcos_etf_adam_s6` | pass | 87 | 0.990539551 | Lower LR, larger readout norm. |
| 6 | `qkv_wcos_etf_muon_v` | pass | 87 | 0.990539551 | Small V Muon residual ablation. |
| 8 | `qkv_wcos_etf_block_beta` | pass | 89 | 0.990295410 | Separate beta2 values by block. |
| 9 | `adamw_fast_synth` | pass | 183 | 0.990478516 | Plain faster AdamW baseline. |
| - | `adamw` | fail | 400 | 0.918029785 | Default AdamW baseline. |
