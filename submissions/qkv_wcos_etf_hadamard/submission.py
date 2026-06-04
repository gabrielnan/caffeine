from __future__ import annotations

from submissions.synthetic_recall import ProjectedAdamW


class Submission(ProjectedAdamW):
    def __init__(self, params):
        super().__init__(params, lr=0.055, beta2=0.9, readout_norm=6.0, frame="hadamard")
