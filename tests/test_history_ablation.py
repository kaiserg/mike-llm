import unittest
import torch
from gpt import GPT
from history_ablation import suffix_isolation_mask, score_history_ablation


class HistoryAblationTests(unittest.TestCase):
    def test_isolation_and_original_positions(self):
        torch.set_num_threads(1)
        for encoding in ('learned', 'sinusoidal'):
            torch.manual_seed(4)
            model = GPT(9, 64, num_layers=2, positional_encoding=encoding).eval()
            ids = torch.randint(0,9,(2,64))
            altered = ids.clone()
            altered[:,:48] = (altered[:,:48]+1)%9
            mask = suffix_isolation_mask(64)
            with torch.no_grad():
                masked = model(ids, attention_mask=mask)[:,48:]
                self.assertTrue(torch.equal(masked,model(altered,attention_mask=mask)[:,48:]))
                self.assertTrue(torch.equal(model(ids), model(ids, attention_mask=torch.ones(64,64,dtype=torch.bool))))
                self.assertFalse(torch.equal(model(ids)[:,-1],model(altered)[:,-1]))
                # Direct suffix at IDs 48..63 is an independent reference.
                table = getattr(model.position_embedding,'weight',None)
                if table is None:
                    table = model.position_embedding.table
                x = model.token_embedding(ids[:,48:]) + table[48:64]
                reference = model.lm_head(model.final_norm(model.blocks(x)))
                torch.testing.assert_close(masked,reference,rtol=1e-5,atol=1e-6)

    def test_restore_state_and_short_history(self):
        model = GPT(9,8).train()
        before = {k:v.clone() for k,v in model.state_dict().items()}
        result = score_history_ablation(model,torch.arange(9)%9,batch_size=3)
        self.assertEqual(result['full'],result['masked'])
        self.assertEqual(result['full']['scored_tokens'],8)
        self.assertTrue(model.training)
        for key,value in model.state_dict().items():
            self.assertTrue(torch.equal(value,before[key]))

    def test_invalid_masks(self):
        model = GPT(9,8)
        ids = torch.zeros(1,8,dtype=torch.long)
        for mask in (torch.ones(8,8),torch.ones(7,7,dtype=torch.bool),torch.zeros(8,8,dtype=torch.bool)):
            with self.assertRaises(ValueError):
                model(ids,attention_mask=mask)
        for keep in (0,-1,True):
            with self.assertRaises(ValueError):
                suffix_isolation_mask(8,keep)

    def test_curve_matches_independent_suffixes(self):
        from history_ablation import score_history_curve
        import math
        import torch.nn.functional as F
        torch.set_num_threads(1)
        for encoding in ('learned','sinusoidal'):
            torch.manual_seed(14)
            model = GPT(9,64,num_layers=2,positional_encoding=encoding).train()
            ids = torch.randint(0,9,(70,))
            result = score_history_curve(model,ids,batch_size=17)
            self.assertTrue(model.training)
            for keep in (8,16,32,48):
                losses = []
                with torch.no_grad():
                    for target in range(1,len(ids)):
                        length = min(target,64)
                        start = max(0,length-keep)
                        suffix = ids[target-length+start:target].unsqueeze(0)
                        positions = torch.arange(start,length)
                        x = model.token_embedding(suffix)+model.position_embedding(positions)
                        logits = model.lm_head(model.final_norm(model.blocks(x)))[:,-1]
                        losses.append(F.cross_entropy(logits.double(),ids[target:target+1]).item())
                self.assertAlmostEqual(result[str(keep)]['mean_nll'],math.fsum(losses)/len(losses),places=6)
                self.assertEqual(result[str(keep)]['scored_tokens'],69)
            # Legacy interface still yields the same 16-token condition.
            legacy = score_history_ablation(model,ids,batch_size=17)
            self.assertEqual(result['16'],legacy['masked'])
            for keeps in ((),(8,8),(0,),(-1,)):
                with self.assertRaises(ValueError):
                    score_history_curve(model,ids,keeps=keeps)
