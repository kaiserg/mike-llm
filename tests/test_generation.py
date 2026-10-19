import unittest

import torch

from generate import generate
from gpt import GPT


class GenerationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = GPT(vocab_size=4, context_length=3)
        self.ids = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 1]])

    def test_sampling_repeatability_and_prompt_preservation(self):
        a = generate(self.model, self.ids, 7, 0.8, torch.Generator().manual_seed(42))
        b = generate(self.model, self.ids, 7, 0.8, torch.Generator().manual_seed(42))
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(a.shape, (2, 12))
        self.assertTrue(torch.equal(a[:, :5], self.ids))
        self.assertFalse(a.requires_grad)
        self.assertTrue(self.model.training)

    def test_greedy_matches_manual_sliding_window(self):
        expected = self.ids.clone()
        with torch.no_grad():
            for _ in range(6):
                token = self.model(expected[:, -3:])[:, -1].argmax(-1, keepdim=True)
                expected = torch.cat((expected, token), dim=1)
        self.model.eval()
        actual = generate(self.model, self.ids, 6, 0)
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(self.model.training)

    def test_explicit_inference_context_matches_manual_short_window(self):
        expected = self.ids.clone()
        with torch.no_grad():
            for _ in range(6):
                token = self.model(expected[:, -2:])[:, -1].argmax(-1, keepdim=True)
                expected = torch.cat((expected, token), dim=1)
        actual = generate(self.model, self.ids, 6, 0, context_length=2)
        self.assertTrue(torch.equal(actual, expected))

    def test_zero_tokens_and_invalid_arguments(self):
        self.assertTrue(torch.equal(generate(self.model, self.ids, 0), self.ids))
        for settings in ({'temperature': -1}, {'temperature': float('nan')},
                         {'temperature': float('inf')}, {'max_new_tokens': -1},
                         {'context_length': 0}, {'context_length': 4}, {'context_length': 1.5}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                generate(self.model, self.ids, **settings)
        for invalid in (self.ids[:, :0], self.ids.float(), self.ids[0], self.ids + 4):
            with self.assertRaises(ValueError):
                generate(self.model, invalid)

    def test_temperature_changes_sampling_distribution(self):
        # A fixed output bias is simulated with constant model logits.
        original = self.model.forward
        def fixed(ids):
            return torch.tensor([0., 1., 2., 3.]).expand(ids.shape[0], ids.shape[1], 4)
        self.model.forward = fixed
        try:
            prompt = torch.zeros(2000, 1, dtype=torch.long)
            cold = generate(self.model, prompt, 1, 0.1, torch.Generator().manual_seed(42))
            hot = generate(self.model, prompt, 1, 5.0, torch.Generator().manual_seed(42))
            self.assertGreater((cold[:, -1] == 3).float().mean(), 0.99)
            self.assertLess((hot[:, -1] == 3).float().mean(), 0.4)
        finally:
            self.model.forward = original


if __name__ == '__main__':
    unittest.main()
