"""Verification loss — plain NLL on log-probabilities, matching KernelGAT.

The model returns ``log(P)`` from its forward pass (see ``KernelKGGPT.forward``),
so ``NLLLoss`` is the right matching head.
"""

import torch.nn as nn


class KernelKGGPTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.nll = nn.NLLLoss()

    def forward(self, outputs, labels):
        loss = self.nll(outputs["logits"], labels)
        return {"total": loss, "verify": loss}
