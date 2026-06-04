from __future__ import annotations

from submissions.synthetic_recall import BlockLrProjectedAdamW


class Submission(BlockLrProjectedAdamW):
    def __init__(self, params):
        super().__init__(
            params,
            switch_step=75,
            qk_late_mult=1.5,
            v_late_mult=0.75,
            w_late_mult=0.5,
        )
