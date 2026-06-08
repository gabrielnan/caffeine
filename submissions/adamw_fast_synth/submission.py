from __future__ import annotations

import torch


class Submission(torch.optim.AdamW):
    def __init__(self, params):
        super().__init__(params, lr=5.0e-2, betas=(0.9, 0.9), weight_decay=0.0)
