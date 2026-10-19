"""Compare every supervised position on the same nonoverlapping validation blocks."""
import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def score_window_grid(model, token_ids, starts, total_length=64, batch_size=128):
    if token_ids.ndim != 1 or token_ids.dtype != torch.long:
        raise ValueError('expected one-dimensional long token IDs')
    if (type(total_length) is not int or total_length < 1 or total_length % model.context_length
            or type(batch_size) is not int or batch_size < 1):
        raise ValueError('total_length must be a positive multiple of model capacity, and batch_size positive')
    if not starts or any(type(s) is not int or s < 0 or s + total_length >= len(token_ids) for s in starts):
        raise ValueError('invalid or empty block starts')
    if len(set(starts)) != len(starts):
        raise ValueError('block starts must be unique')
    sums = torch.zeros(total_length, dtype=torch.double)
    correct = torch.zeros(total_length, dtype=torch.long)
    was_training = model.training
    model.eval()
    try:
        for begin in range(0, len(starts), batch_size):
            block_starts = starts[begin:begin+batch_size]
            for offset in range(0, total_length, model.context_length):
                length = model.context_length
                x = torch.stack([token_ids[s+offset:s+offset+length] for s in block_starts])
                y = torch.stack([token_ids[s+offset+1:s+offset+length+1] for s in block_starts])
                logits = model(x)
                losses = F.cross_entropy(logits.double().reshape(-1,logits.shape[-1]),y.reshape(-1),reduction='none').reshape(y.shape)
                if not torch.isfinite(losses).all():
                    raise RuntimeError('nonfinite window loss')
                sums[offset:offset+length] += losses.sum(0).cpu()
                correct[offset:offset+length] += (logits.argmax(-1)==y).sum(0).cpu()
    finally:
        model.train(was_training)
    positions = [dict(block_column=i, position_id=i % model.context_length,
                      history_length=i % model.context_length+1,
                      mean_nll=sums[i].item()/len(starts), accuracy=correct[i].item()/len(starts),
                      scored_targets=len(starts)) for i in range(total_length)]
    return dict(blocks=len(starts), scored_targets=len(starts)*total_length,
                mean_nll=math.fsum(p['mean_nll'] for p in positions)/total_length,
                positions=positions)
