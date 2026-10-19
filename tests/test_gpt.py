import unittest

import torch
import torch.nn.functional as F

from gpt import GPT


class GPTTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = GPT(vocab_size=7, context_length=6)
        self.ids = torch.randint(0, 7, (2, 6))

    def test_prefix_causality_and_batch_independence(self):
        with torch.no_grad():
            logits = self.model(self.ids)
            changed = self.ids.clone()
            changed[:, 3:] = (changed[:, 3:] + 1) % 7
            torch.testing.assert_close(self.model(changed)[:, :3], logits[:, :3])
            torch.testing.assert_close(self.model(self.ids[:, :3]), logits[:, :3])
            changed[1] = (changed[1] + 2) % 7
            torch.testing.assert_close(self.model(changed)[0, :3], logits[0, :3])

    def test_shifted_loss_reaches_all_parameters(self):
        logits = self.model(self.ids[:, :-1])
        self.assertEqual(logits.shape, (2, 5, 7))
        loss = F.cross_entropy(logits.reshape(-1, 7), self.ids[:, 1:].reshape(-1))
        loss.backward()
        for name, parameter in self.model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_layers_and_embedding_head_are_independent(self):
        self.assertIsNot(self.model.blocks[0], self.model.blocks[1])
        first = next(self.model.blocks[0].parameters())
        second = next(self.model.blocks[1].parameters())
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        self.assertNotEqual(self.model.token_embedding.weight.data_ptr(), self.model.lm_head.weight.data_ptr())

    def test_input_boundaries(self):
        self.assertEqual(self.model(self.ids[:, :1]).shape, (2, 1, 7))
        self.assertEqual(self.model(self.ids).shape, (2, 6, 7))
        for invalid in (torch.zeros(2, 7, dtype=torch.long), self.ids[0], self.ids.float(), self.ids[:, :0], self.ids[:0]):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                self.model(invalid)

    def test_invalid_configuration(self):
        for config in ({'num_layers': 0}, {'num_heads': 5}, {'context_length': -1}, {'expansion_factor': 1.5}):
            args = dict(vocab_size=7, context_length=6)
            args.update(config)
            with self.subTest(config=config), self.assertRaises(ValueError):
                GPT(**args)


if __name__ == '__main__':
    unittest.main()
