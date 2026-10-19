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
