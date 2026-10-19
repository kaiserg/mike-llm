"""Remove distant token content while retaining each window's position IDs."""
import math
import torch
import torch.nn.functional as F


def suffix_isolation_mask(length, keep=16, device=None):
    """Suffix queries cannot read the prefix, in any layer; True permits attention."""
    if type(length) is not int or length < 1 or type(keep) is not int or keep < 1:
        raise ValueError('length and keep must be positive integers')
    positions = torch.arange(length, device=device)
    cutoff = max(0, length - keep)
    return (positions[:, None] < cutoff) | (positions[None, :] >= cutoff)


@torch.no_grad()
def score_history_curve(model, ids, batch_size=128, keeps=(8, 16, 32, 48)):
    """Score each target at several isolated suffix lengths and full history."""
    if ids.ndim != 1 or ids.dtype != torch.long or len(ids) < 2:
        raise ValueError('ids must be a torch.long vector with at least two tokens')
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be positive')
    if not keeps or len(set(keeps)) != len(keeps):
        raise ValueError('keeps must be nonempty and unique')
    for keep in keeps:
        suffix_isolation_mask(1, keep)
    conditions = {'full': None, **{str(k): k for k in keeps}}
    totals = {name: dict(losses=[], correct=0) for name in conditions}
    was_training = model.training
    model.eval()
    try:
        for start in range(1, len(ids), batch_size):
            groups = {}
            for target in range(start, min(start + batch_size, len(ids))):
                groups.setdefault(min(target, model.context_length), []).append(target)
            for length, targets in groups.items():
                inputs = torch.stack([ids[t-length:t] for t in targets])
                for name, keep in conditions.items():
                    mask = None if keep is None else suffix_isolation_mask(length, keep, ids.device)
                    logits = model(inputs, attention_mask=mask)[:, -1]
                    losses = F.cross_entropy(logits.double(), ids[targets], reduction='none')
                    if not torch.isfinite(losses).all():
                        raise RuntimeError('nonfinite ablation loss')
                    totals[name]['losses'].extend(losses.tolist())
                    totals[name]['correct'] += (logits.argmax(-1) == ids[targets]).sum().item()
    finally:
        model.train(was_training)
    result = {}
    for name, total in totals.items():
        mean = math.fsum(total['losses']) / (len(ids)-1)
        result[name] = dict(scored_tokens=len(ids)-1, mean_nll=mean,
                            perplexity=math.exp(mean), accuracy=total['correct']/(len(ids)-1))
    return result


def score_history_ablation(model, ids, batch_size=128, keep=16):
    """Backward-compatible two-condition evaluation."""
    curve = score_history_curve(model, ids, batch_size, keeps=(keep,))
    return dict(full=curve['full'], masked=curve[str(keep)],
                masked_minus_full_nll=curve[str(keep)]['mean_nll'] - curve['full']['mean_nll'])
