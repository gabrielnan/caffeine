from __future__ import annotations

from submissions.synthetic_recall import BlockBetaProjectedAdamW


class Submission(BlockBetaProjectedAdamW):
    def __init__(self, params):
        super().__init__(params, qk_beta2=0.8, v_beta2=0.95, w_beta2=0.95)
