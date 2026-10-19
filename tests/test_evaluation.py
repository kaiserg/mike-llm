import math
import unittest

import torch
import torch.nn.functional as F

from evaluate import evaluate_text
from gpt import GPT
from tokenizer import CharacterTokenizer


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.tokenizer = CharacterTokenizer('abc')
        self.model = GPT(vocab_size=3, context_length=3)
        self.text = 'abcabacbca'

    def test_matches_manual_sliding_window_and_preserves_model(self):
        original = {k: v.clone() for k, v in self.model.state_dict().items()}
        ids = torch.tensor(self.tokenizer.encode(self.text))
        losses = []
        with torch.no_grad():
            for target in range(1, len(ids)):
                logits = self.model(ids[max(0, target - 3):target][None])[:, -1]
                losses.append(F.cross_entropy(logits.double(), ids[target:target + 1]).item())
        result = evaluate_text(self.model, self.tokenizer, self.text, batch_size=4)
        self.assertEqual(result['scored_tokens'], len(self.text) - 1)
        self.assertAlmostEqual(result['mean_nll'], sum(losses) / len(losses), places=6)
        one = evaluate_text(self.model, self.tokenizer, self.text, batch_size=1)
        self.assertAlmostEqual(result['mean_nll'], one['mean_nll'], places=6)
        self.assertTrue(self.model.training)
        for key, value in original.items():
            self.assertTrue(torch.equal(value, self.model.state_dict()[key]))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_context_cap_matches_manual_short_windows(self):
        ids = torch.tensor(self.tokenizer.encode(self.text))
        losses = []
        with torch.no_grad():
            for target in range(1, len(ids)):
                logits = self.model(ids[max(0, target - 2):target][None])[:, -1]
                losses.append(F.cross_entropy(logits.double(), ids[target:target + 1]).item())
        report = evaluate_text(self.model, self.tokenizer, self.text, context_length=2)
        self.assertAlmostEqual(report['mean_nll'], sum(losses) / len(losses), places=6)
        self.assertEqual(report['scored_tokens'], len(self.text) - 1)
        for invalid in (0, 4, -1, 1.5):
            with self.assertRaises(ValueError):
                evaluate_text(self.model, self.tokenizer, self.text, context_length=invalid)

    def test_uniform_logits_have_vocabulary_perplexity(self):
        with torch.no_grad():
            self.model.lm_head.weight.zero_()
        result = evaluate_text(self.model, self.tokenizer, self.text)
        self.assertAlmostEqual(result['mean_nll'], math.log(3))
        self.assertAlmostEqual(result['perplexity'], 3.)
        self.assertAlmostEqual(result['bits_per_character'], math.log2(3))
        self.assertAlmostEqual(result['next_token_accuracy'], self.text[1:].count('a') / 9)

    def test_short_text_and_errors_restore_mode(self):
        self.model.eval()
        self.assertEqual(evaluate_text(self.model, self.tokenizer, 'ab')['scored_tokens'], 1)
        self.assertFalse(self.model.training)
        for text in ('', 'a', 'abx'):
            with self.assertRaises(ValueError):
                evaluate_text(self.model, self.tokenizer, text)
        with self.assertRaises(ValueError):
            evaluate_text(self.model, self.tokenizer, 'ab', batch_size=0)
        self.model.train()
        def broken(inputs):
            raise RuntimeError('test forward failure')
        self.model.forward = broken
        with self.assertRaises(RuntimeError):
            evaluate_text(self.model, self.tokenizer, 'ab')
        self.assertTrue(self.model.training)


if __name__ == '__main__':
    unittest.main()
