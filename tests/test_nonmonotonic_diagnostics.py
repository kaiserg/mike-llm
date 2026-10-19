import unittest
import torch
from gpt import GPT
from history_ablation import suffix_isolation_mask
from nonmonotonic_diagnostics import forward_layer_histories


class NonmonotonicTests(unittest.TestCase):
    def test_endpoints_mixed_isolation_and_attention(self):
        torch.set_num_threads(1)
        for encoding in ('learned','sinusoidal'):
            torch.manual_seed(5)
            model=GPT(9,64,positional_encoding=encoding).eval()
            ids=torch.randint(0,9,(2,64))
            before={k:v.clone() for k,v in model.state_dict().items()}
            for keeps in ((8,8),(16,8),(8,16),(16,16)):
                logits,stats=forward_layer_histories(model,ids,keeps)
                self.assertEqual(stats.shape,(2,2,3,2))
                self.assertTrue(torch.isfinite(stats).all())
                changed=ids.clone();changed[:,:48]=(changed[:,:48]+1)%9
                other,_=forward_layer_histories(model,changed,keeps)
                self.assertTrue(torch.equal(logits,other))
                for layer,keep in enumerate(keeps):
                    if keep==8:self.assertTrue((stats[:,layer,:,0]==0).all())
                    self.assertTrue((stats[:,layer,:,0]>=0).all())
                    self.assertTrue((stats[:,layer,:,0]<=1).all())
                if keeps[0]==keeps[1]:
                    with torch.no_grad():
                        native=model(ids,attention_mask=suffix_isolation_mask(64,keeps[0]))[:,-1]
                    self.assertTrue(torch.equal(logits,native))
            for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,before[k]))
            for block in model.blocks:
                for head in block.attention.heads:self.assertFalse(head._forward_pre_hooks)

    def test_reversal_changes_only_added_content_and_short_prefixes(self):
        model=GPT(9,64).eval()
        ids=torch.arange(64).remainder(9).unsqueeze(0)
        expected=ids.clone();expected[:,48:56]=expected[:,48:56].flip(1)
        a,_=forward_layer_histories(model,ids,(16,16),reverse_added=True)
        b,_=forward_layer_histories(model,expected,(16,16))
        self.assertTrue(torch.equal(a,b))
        for length in (1,4,8,12,16):
            a,_=forward_layer_histories(model,ids[:,:length],(16,16))
            with torch.no_grad():b=model(ids[:,:length])[:,-1]
            self.assertTrue(torch.equal(a,b))
        with self.assertRaises(ValueError):forward_layer_histories(model,ids,(8,))
