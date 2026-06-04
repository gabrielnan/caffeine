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
