from typing import Dict

import torch

@torch.no_grad()
def binary_metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5) -> Dict[str, float]:
    """
    logits: [B] or [B,1]
    targets: [B] float {0,1}
    """
    logits = logits.view(-1)
    targets = targets.view(-1)
    probs = torch.sigmoid(logits)
    preds = (probs >= thr).to(targets.dtype)

    tp = ((preds == 1) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()

    eps = 1e-9
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return {"acc": acc, "precision": precision, "recall": recall, "f1": f1}
