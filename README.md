# caffeine: attention optimization problem

**Motivation:** most gradient optimizers are agnostic to values being optimized.
Muon has demonstrated that matrix-specific optimizer can outperform a fully agnostic one.
The goal of this task is to see whether we can create an even better optimizer by finding the best optimizer specifically for attention models.

Task: design an optimizer that trains a fixed vanilla self-attention model to match a deterministic teacher from input/output samples.


## fixed values

1. dataset and train/val/eval splits
2. model architecture

## submission contract

1. A submission is a Python file defining `Submission`, a subclass of `torch.optim.Optimizer`.
2. The harness instantiates it strictly as `Submission(model.parameters())`.
3. The harness calls `optimizer.step()` without a closure; optimizers must update parameters using the `.grad` fields populated by the harness.

## scoring

Official v0 scoring is wall-clock time on GitHub Actions `macos-15` arm64. The harness uses MPS when available and falls back to CPU. It stops at the first fixed evaluation checkpoint where the selected track's target metric is met; submissions that miss the target within that track's step budget fail.

Run locally:

```bash
uv sync
uv run python test_benchmark.py
uv run python run_eval.py --submission submissions/adamw/submission.py --results-json result.json
```

## dataset

Each track deterministically initializes its own model, train data, eval data, and fixed stochastic batch order from public config values in `tracks/`. For `random_teacher`, train and eval targets are teacher outputs on random matrix inputs. For token-recall tracks, targets are value classes from the generated key/value bindings.

## tracks

The benchmark supports multiple tracks through `run_eval.py --track`:

| Track | Objective | Target |
| --- | --- | --- |
| `random_teacher` | Student attention matches a deterministic random teacher attention model on random matrix inputs. | eval MSE <= `4e-7` |
| `single_ar` | Opaque single-query associative recall: `8` pair tokens + `1` query token -> value class. | eval accuracy >= `0.99` |
| `mqar` | Opaque multi-query associative recall: `8` pair tokens + `8` query tokens -> value classes. | eval accuracy >= `0.99` |

`mqar` is inspired by the multi-query associative recall task from
[Zoology: Measuring and Improving Recall in Efficient Language Models](https://arxiv.org/abs/2312.04927).

All tracks keep the same optimizer contract: submissions define
`Submission(torch.optim.Optimizer)`, and the harness instantiates it as
`Submission(model.parameters())`.

Run a specific track:

```bash
uv run python run_eval.py --track single_ar --submission submissions/adamw/submission.py
uv run python run_eval.py --track mqar --submission submissions/adamw/submission.py
```

## track architecture

Each benchmark track implements the `BenchmarkTrack` protocol in `tracks/base.py`.
The runner asks the selected track for only the shared benchmark surface:
student model construction, train/eval datasets, batch indices, loss,
evaluation, target metrics, metric updates, `max_steps`, and `eval_every`.
Shared data-loop settings live in `RunConfig` as a helper for track
implementations; dataset-specific fields live in each track module's own config
dataclass. This keeps track-specific details out of `run_eval.py` without forcing
unrelated tracks to share one large task config.

`tracks/random_teacher.py` implements the original random-teacher task with
single-head vanilla self-attention via `torch.nn.MultiheadAttention`.
`tracks/token_recall.py` implements the synthetic recall tasks with an equivalent
single-head token-attention classifier using trainable Q/K/V token tables and a
linear readout.
