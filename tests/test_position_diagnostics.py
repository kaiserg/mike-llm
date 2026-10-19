import math
import unittest

import torch
import torch.nn.functional as F

from gpt import GPT
from position_diagnostics import score_positions
from window_diagnostics import score_window_grid


class PositionDiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = GPT(vocab_size=4, context_length=8)
        self.ids = torch.tensor([0, 1, 2, 3, 0, 3, 2, 1, 0, 1, 3, 2])
        self.targets = [8, 10, 11]

    def test_same_targets_match_manual_prefix_scoring(self):
        result = score_positions(self.model, self.ids, self.targets, [1, 3, 8], batch_size=2)
        for row in result:
            length = row['history_length']
            losses = []
            with torch.no_grad():
                for target in self.targets:
                    logits = self.model(self.ids[target-length:target][None])[:, -1]
                    losses.append(F.cross_entropy(logits.double(), self.ids[target:target+1]).item())
            self.assertAlmostEqual(row['mean_nll'], math.fsum(losses)/3, places=6)
            self.assertEqual(row['scored_targets'], 3)
        self.assertTrue(self.model.training)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_shift_matches_direct_embedding_computation_and_restores_state(self):
        before = {k:v.clone() for k,v in self.model.state_dict().items()}
        row = score_positions(self.model, self.ids, self.targets, [3], position_offset=5)[0]
        inputs = torch.stack([self.ids[t-3:t] for t in self.targets])
        with torch.no_grad():
            x = self.model.token_embedding(inputs) + self.model.position_embedding.weight[5:8]
            logits = self.model.lm_head(self.model.final_norm(self.model.blocks(x)))[:, -1]
            expected = F.cross_entropy(logits.double(), self.ids[self.targets]).item()
        self.assertAlmostEqual(row['mean_nll'], expected, places=6)
        self.assertEqual(len(self.model.position_embedding._forward_hooks), 0)
        for key,value in before.items():
            self.assertTrue(torch.equal(value,self.model.state_dict()[key]))

    def test_window_columns_match_manual_reset_prefixes(self):
        model = GPT(vocab_size=4, context_length=4)
        starts = [0, 3]
        result = score_window_grid(model,self.ids,starts,total_length=8,batch_size=1)
        for column,row in enumerate(result['positions']):
            losses=[]
            for start in starts:
                chunk_start=start+(column//4)*4
                target=start+column+1
                with torch.no_grad():
                    logits=model(self.ids[chunk_start:target][None])[:,-1]
                    losses.append(F.cross_entropy(logits.double(),self.ids[target:target+1]).item())
            self.assertAlmostEqual(row['mean_nll'],sum(losses)/2,places=6)
            self.assertEqual(row['position_id'],column%4)
        self.assertEqual(result['scored_targets'],16)
        self.assertAlmostEqual(result['mean_nll'],sum(p['mean_nll'] for p in result['positions'])/8)
        self.assertTrue(model.training)
        with self.assertRaises(ValueError):
            score_window_grid(model,self.ids,[0],total_length=7)

    def test_errors_and_exception_cleanup(self):
        for lengths,offset in [([9],0),([3],6),([0],0)]:
            with self.assertRaises(ValueError):
                score_positions(self.model,self.ids,self.targets,lengths,position_offset=offset)
        with self.assertRaises(ValueError):
            score_positions(self.model,self.ids,[1],[3])
        def fail(inputs):
            self.model.position_embedding(torch.arange(inputs.shape[1]))
            raise RuntimeError('test failure')
        self.model.forward = fail
        with self.assertRaises(RuntimeError):
            score_positions(self.model,self.ids,self.targets,[3],position_offset=5)
        self.assertTrue(self.model.training)
        self.assertEqual(len(self.model.position_embedding._forward_hooks),0)


if __name__ == '__main__':
    unittest.main()
