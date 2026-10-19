"""Score identical validation targets at controlled history lengths and position IDs."""
import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def score_positions(model, token_ids, targets, lengths, batch_size=128, position_offset=0):
    """Return last-position loss by prefix length, using the same target indices.

    A nonzero offset shifts all learned position IDs in a prefix without changing
    token history. It is a diagnostic intervention, not the native GPT forward.
    """
    if token_ids.ndim != 1 or token_ids.dtype != torch.long:
        raise ValueError('token_ids must be a one-dimensional torch.long tensor')
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be positive')
    if not lengths or any(type(n) is not int or n < 1 for n in lengths):
        raise ValueError('lengths must contain positive integers')
    if type(position_offset) is not int or position_offset < 0:
        raise ValueError('position_offset must be nonnegative')
    if max(lengths) + position_offset > model.context_length:
        raise ValueError('history plus position offset exceeds model capacity')
    if not targets or any(type(t) is not int or t < max(lengths) or t >= len(token_ids) for t in targets):
        raise ValueError('targets must be valid indices with enough preceding history')
    if len(set(targets)) != len(targets):
        raise ValueError('target indices must be unique')
    was_training = model.training
    model.eval()
    hook = None
    try:
        if position_offset:
            def shifted(module, args, output):
                return module.weight[args[0] + position_offset]
            hook = model.position_embedding.register_forward_hook(shifted)
        rows = []
        for length in lengths:
            losses, correct = [], 0
            for start in range(0, len(targets), batch_size):
                batch = targets[start:start + batch_size]
                inputs = torch.stack([token_ids[t - length:t] for t in batch])
                labels = token_ids[batch]
                logits = model(inputs)[:, -1]
                nll = F.cross_entropy(logits.double(), labels, reduction='none')
                if not torch.isfinite(nll).all():
                    raise RuntimeError('nonfinite position loss')
                losses.extend(nll.tolist())
                correct += (logits.argmax(-1) == labels).sum().item()
            mean = math.fsum(losses) / len(targets)
            rows.append(dict(history_length=length, final_position_id=position_offset + length - 1,
                             position_offset=position_offset, scored_targets=len(targets),
                             mean_nll=mean, perplexity=math.exp(mean), accuracy=correct / len(targets)))
        return rows
    finally:
        if hook is not None:
            hook.remove()
        model.train(was_training)
