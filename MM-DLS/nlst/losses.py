"""Masked losses for independently observed prognosis endpoints."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _binary_mask(mask, *, device, shape):
    mask = torch.as_tensor(mask, device=device)
    if tuple(mask.shape) != tuple(shape):
        raise ValueError("mask shape must match target shape")
    if not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all():
        raise ValueError("mask must contain only 0 or 1")
    return mask.bool()


def masked_bce(logits, targets, mask):
    """Mean BCE over observed cells; masked NaN placeholders are safe.

    An empty observation set returns a differentiable zero.
    """
    if not torch.is_floating_point(logits):
        raise ValueError("logits must have floating point dtype")
    targets = torch.as_tensor(targets, dtype=logits.dtype, device=logits.device)
    if targets.shape != logits.shape:
        raise ValueError("targets and logits must have identical shapes")
    observed = _binary_mask(mask, device=logits.device, shape=logits.shape)
    selected_logits, selected_targets = logits[observed], targets[observed]
    if not observed.any():
        return selected_logits.sum()
    if not torch.isfinite(selected_logits).all():
        raise ValueError("Observed logits must be finite")
    if not torch.isfinite(selected_targets).all() or not ((selected_targets == 0) | (selected_targets == 1)).all():
        raise ValueError("Observed BCE targets must contain only 0 or 1")
    return F.binary_cross_entropy_with_logits(selected_logits, selected_targets)


def cox_breslow(risk, time, event, validmask=None):
    """Mean negative Cox partial log-likelihood with Breslow event ties.

    Every event at t uses the denominator over ALL patients observed through t,
    including patients censored at t. Missing endpoint rows are removed before
    numerical operations. A batch with no events returns differentiable zero.
    """
    if not torch.is_floating_point(risk):
        raise ValueError("risk must have floating point dtype")
    risk = risk.reshape(-1)
    time = torch.as_tensor(time, device=risk.device).reshape(-1)
    event = torch.as_tensor(event, device=risk.device).reshape(-1)
    if not (risk.shape == time.shape == event.shape):
        raise ValueError("risk, time, and event must have identical lengths")
    valid = torch.ones_like(event, dtype=torch.bool) if validmask is None else _binary_mask(
        torch.as_tensor(validmask, device=risk.device).reshape(-1), device=risk.device, shape=risk.shape
    )
    risk, time, event = risk[valid], time[valid], event[valid]
    if not torch.isfinite(risk).all() or not torch.isfinite(time).all() or (time < 0).any():
        raise ValueError("Observed risks/times must be finite and times nonnegative")
    if not torch.isfinite(event).all() or not ((event == 0) | (event == 1)).all():
        raise ValueError("Observed event indicators must contain only 0 or 1")
    if not event.eq(1).any():
        return risk[:0].sum()
    if risk.dtype in (torch.float16, torch.bfloat16):
        risk = risk.float()
    order = torch.argsort(time, descending=True)
    risk, time, event = risk[order], time[order], event[order]
    _, counts = torch.unique_consecutive(time, return_counts=True)
    group_ends = counts.cumsum(0) - 1
    group_denominators = torch.logcumsumexp(risk, dim=0)[group_ends]
    denominator_per_row = torch.repeat_interleave(group_denominators, counts)
    events = event.eq(1)
    return (denominator_per_row[events] - risk[events]).mean()

