"""Inference-only layer interventions for fixed-position history diagnostics."""
import torch
from history_ablation import suffix_isolation_mask


@torch.no_grad()
def forward_layer_histories(model, token_ids, keeps, reverse_added=False):
    """Use separate isolated suffix masks per block and observe final-query attention.

    Attention mass is descriptive, not an attribution of output importance.
    reverse_added reverses only the eight tokens immediately preceding the latest
    eight; all positions and latest tokens remain unchanged.
    """
    if len(keeps) != len(model.blocks):
        raise ValueError('one history length is required per block')
    length = token_ids.shape[1]
    if reverse_added:
        token_ids = token_ids.clone()
        start, stop = max(0,length-16), max(0,length-8)
        token_ids[:,start:stop] = token_ids[:,start:stop].flip(1)
    positions = torch.arange(length,device=token_ids.device)
    x = model.token_embedding(token_ids)+model.position_embedding(positions)
    observations = []
    for block, keep in zip(model.blocks, keeps):
        permission = (suffix_isolation_mask(length,16,x.device)
                      & suffix_isolation_mask(length,keep,x.device))
        handles, head_rows = [], []
        def observe(head,args,kwargs):
            inputs = args[0]
            q = head.query(inputs[:,-1:])
            keys = head.key(inputs)
            scores = (q @ keys.transpose(-2,-1)) / keys.shape[-1]**0.5
            weights = torch.softmax(scores.masked_fill(~permission[-1],float('-inf')),dim=-1)[:,0]
            start,stop=max(0,length-16),max(0,length-8)
            head_rows.append(torch.stack((weights[:,start:stop].sum(-1),
                -(weights*weights.clamp_min(1e-30).log()).sum(-1)),dim=-1))
        try:
            for head in block.attention.heads:
                handles.append(head.register_forward_pre_hook(observe,with_kwargs=True))
            x = block(x,attention_mask=permission)
        finally:
            for handle in handles:
                handle.remove()
        observations.append(torch.stack(head_rows,dim=1))
    return model.lm_head(model.final_norm(x))[:,-1], torch.stack(observations,dim=1)
