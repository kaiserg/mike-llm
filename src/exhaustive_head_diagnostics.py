"""Cached exact-computation structure for the six-head 8/16 mask cube."""
import torch
from history_ablation import suffix_isolation_mask


def blocked_pairs(mask):
    if type(mask) is not int or not 0 <= mask < 64:
        raise ValueError('mask must be an integer from 0 to 63')
    return [(bit//3,bit%3) for bit in range(6) if mask & (1<<bit)]


@torch.no_grad()
def forward_all_head_masks(model, token_ids):
    """Return [batch, 64 masks, vocabulary]; bit 1 means isolate latest eight.

    Retain original IDs when dropping the already-isolated older prefix. Cache
    head outputs for both masks; second-block caches depend on first-block mask.
    No model parameters, buffers or hooks are modified.
    """
    if len(model.blocks)!=2 or any(len(b.attention.heads)!=3 for b in model.blocks):
        raise ValueError('expected two blocks with three heads each')
    if token_ids.ndim!=2 or token_ids.dtype!=torch.long or not 1<=token_ids.shape[1]<=model.context_length:
        raise ValueError('invalid token tensor or sequence length')
    length=token_ids.shape[1]
    offset=max(0,length-16)
    token_ids=token_ids[:,offset:]
    positions=torch.arange(offset,length,device=token_ids.device)
    x=model.token_embedding(token_ids)+model.position_embedding(positions)
    masks={k:suffix_isolation_mask(token_ids.shape[1],k,token_ids.device) for k in (8,16)}
    def choices(block,state):
        normalized=block.norm1(state)
        return [[head(normalized,attention_mask=masks[k]) for k in (16,8)] for head in block.attention.heads]
    def finish(block,state,outputs,bits):
        merged=torch.cat([outputs[h][(bits>>h)&1] for h in range(3)],dim=-1)
        state=state+block.attention.projection(merged)
        return state+block.feed_forward(block.norm2(state))
    first=model.blocks[0];second=model.blocks[1]
    first_choices=choices(first,x)
    logits=[None]*64
    for lower in range(8):
        state=finish(first,x,first_choices,lower)
        second_choices=choices(second,state)
        for upper in range(8):
            final=finish(second,state,second_choices,upper)
            logits[lower+(upper<<3)]=model.lm_head(model.final_norm(final))[:,-1]
    return torch.stack(logits,dim=1)


def analyze_mask_cube(nll):
    """Summarize exact conditional edges, square interactions and Shapley gains."""
    import math
    import statistics
    if len(nll)!=64:raise ValueError('expected 64 losses')
    gains=[nll[0]-x for x in nll]
    shapley={};edges={}
    for bit in range(6):
        marginal=[gains[s|(1<<bit)]-gains[s] for s in range(64) if not s&(1<<bit)]
        weights=[math.factorial(s.bit_count())*math.factorial(5-s.bit_count())/math.factorial(6)
                 for s in range(64) if not s&(1<<bit)]
        name=f'block{bit//3+1}_head{bit%3+1}'
        shapley[name]=math.fsum(w*v for w,v in zip(weights,marginal))
        edges[name]=dict(min_gain=min(marginal),max_gain=max(marginal),
            mean_gain=statistics.mean(marginal),positive_contexts=sum(v>0 for v in marginal),
            negative_contexts=sum(v<0 for v in marginal),context_count=32,
            individual_gain=gains[1<<bit],gain_given_all_other_masks=gains[63]-gains[63^(1<<bit)])
    interactions=[]
    for a in range(6):
        for b in range(a+1,6):
            values=[gains[s|(1<<a)|(1<<b)]-gains[s|(1<<a)]-gains[s|(1<<b)]+gains[s]
                    for s in range(64) if not s&((1<<a)|(1<<b))]
            interactions.append(dict(bits=[a,b],context_count=16,mean=statistics.mean(values),
                minimum=min(values),maximum=max(values),positive_contexts=sum(v>0 for v in values)))
    # Möbius coefficients recover gain for every subset, including higher orders.
    coefficients=gains.copy()
    for bit in range(6):
        for s in range(64):
            if s&(1<<bit):coefficients[s]-=coefficients[s^(1<<bit)]
    order_summary=[]
    for order in range(1,7):
        values=[coefficients[s] for s in range(64) if s.bit_count()==order]
        order_summary.append(dict(order=order,count=len(values),sum=math.fsum(values),
                                  sum_absolute=math.fsum(abs(v) for v in values)))
    return dict(shapley_gains=shapley,conditional_edges=edges,pair_interactions=interactions,
                mobius_gain_coefficients=coefficients,mobius_by_order=order_summary)
