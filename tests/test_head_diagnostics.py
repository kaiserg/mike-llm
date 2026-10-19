import unittest
import torch
from gpt import GPT
from head_diagnostics import forward_head_masks
from history_ablation import suffix_isolation_mask
from nonmonotonic_diagnostics import forward_layer_histories


class HeadDiagnosticTests(unittest.TestCase):
    def test_controls_and_selective_hook_reference(self):
        torch.set_num_threads(1)
        for encoding in ('learned','sinusoidal'):
            torch.manual_seed(9)
            model=GPT(9,64,positional_encoding=encoding).eval()
            ids=torch.randint(0,9,(2,64))
            before={k:v.clone() for k,v in model.state_dict().items()}
            conditions=(((),(16,16)), ([(0,h) for h in range(3)],(8,16)),
                ([(1,h) for h in range(3)],(16,8)), ([(b,h) for b in range(2) for h in range(3)],(8,8)))
            for blocked,keeps in conditions:
                a=forward_head_masks(model,ids,blocked)
                b,_=forward_layer_histories(model,ids,keeps)
                self.assertTrue(torch.equal(a,b))
            # Independent path through native GPT, overriding only one head's mask.
            for block in range(2):
                for head in range(3):
                    def restrict(module,args,kwargs):
                        return args,{**kwargs,'attention_mask':suffix_isolation_mask(64,16)&suffix_isolation_mask(64,8)}
                    handle=model.blocks[block].attention.heads[head].register_forward_pre_hook(restrict,with_kwargs=True)
                    try:
                        with torch.no_grad():reference=model(ids,attention_mask=suffix_isolation_mask(64,16))[:,-1]
                    finally:handle.remove()
                    actual=forward_head_masks(model,ids,[(block,head)])
                    self.assertTrue(torch.equal(actual,reference))
                    changed=ids.clone();changed[:,:48]=(changed[:,:48]+1)%9
                    self.assertTrue(torch.equal(actual,forward_head_masks(model,changed,[(block,head)])))
            for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,before[k]))
            for block in model.blocks:
                for head in block.attention.heads:self.assertFalse(head._forward_pre_hooks)

    def test_short_windows_and_invalid_pair(self):
        model=GPT(9,64).eval()
        for length in (1,4,8):
            ids=torch.randint(0,9,(2,length))
            self.assertTrue(torch.equal(forward_head_masks(model,ids),forward_head_masks(model,ids,[(0,0)])))
        with self.assertRaises(ValueError):forward_head_masks(model,ids,[(2,0)])

    def test_joint_masks_against_native_path(self):
        torch.set_num_threads(1)
        # Exactly the previously selected sets and each leave-one-out variant.
        selected_sets=(((0,0),(1,0),(1,1),(1,2)),
                       ((1,0),(1,2),(1,1),(0,2),(0,0)))
        for encoding,selected in zip(('learned','sinusoidal'),selected_sets):
            torch.manual_seed(17)
            model=GPT(9,64,positional_encoding=encoding).eval()
            ids=torch.randint(0,9,(2,64))
            groups=[selected]+[tuple(p for p in selected if p!=removed) for removed in selected]
            for blocked in groups:
                handles=[]
                def restrict(module,args,kwargs):
                    return args,{**kwargs,'attention_mask':suffix_isolation_mask(64,16)&suffix_isolation_mask(64,8)}
                try:
                    for block,head in blocked:
                        handles.append(model.blocks[block].attention.heads[head].register_forward_pre_hook(restrict,with_kwargs=True))
                    with torch.no_grad():reference=model(ids,attention_mask=suffix_isolation_mask(64,16))[:,-1]
                finally:
                    for handle in handles:handle.remove()
                actual=forward_head_masks(model,ids,blocked)
                self.assertTrue(torch.equal(actual,reference))
                perturbed=ids.clone();perturbed[:,:48]=(perturbed[:,:48]+1)%9
                self.assertTrue(torch.equal(actual,forward_head_masks(model,perturbed,blocked)))
                for block in model.blocks:
                    for head in block.attention.heads:self.assertFalse(head._forward_pre_hooks)
