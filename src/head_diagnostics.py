"""Per-head suffix isolation with unchanged tokens, positions and weights."""
import torch
from history_ablation import suffix_isolation_mask


@torch.no_grad()
def forward_head_masks(model, token_ids, blocked=()):
    """Block the added eight for selected (zero-based block, head) pairs.

    Every head is confined to the final sixteen first. A selected head further
    isolates the latest eight at every query, not just the final query.
    """
    blocked=set(blocked)
    valid={(b,h) for b,block in enumerate(model.blocks) for h in range(len(block.attention.heads))}
    if not blocked <= valid:
        raise ValueError('unknown block/head pair')
    length=token_ids.shape[1]
    masks={k:suffix_isolation_mask(length,k,token_ids.device) for k in (8,16)}
    positions=torch.arange(length,device=token_ids.device)
    x=model.token_embedding(token_ids)+model.position_embedding(positions)
    for b,block in enumerate(model.blocks):
        normalized=block.norm1(x)
        outputs=[head(normalized,attention_mask=masks[16] & masks[8] if (b,h) in blocked else masks[16])
                 for h,head in enumerate(block.attention.heads)]
        x=x+block.attention.projection(torch.cat(outputs,dim=-1))
        x=x+block.feed_forward(block.norm2(x))
    return model.lm_head(model.final_norm(x))[:,-1]
