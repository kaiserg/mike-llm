import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import torch

from gpt import GPT
from train import DEMO_TRAIN, DEMO_VALIDATION, prepare_data, run_training, sample_batch


class TrainingTests(unittest.TestCase):
    def test_shift_and_split(self):
        _, train, validation = prepare_data("abcabcabc", "cbacbacba", 3)
        x, y = sample_batch(train, 4, 3, torch.Generator().manual_seed(1))
        torch.testing.assert_close(x[:, 1:], y[:, :-1])
        self.assertEqual(validation.numel(), 9)
        with self.assertRaises(ValueError):
            prepare_data("abcabc", "xyzxyz", 3)
        with self.assertRaises(ValueError):
            prepare_data("abc", "abcabc", 3)

    def test_step_observer_preserves_training_trajectory(self):
        observed = []
        def observer(step, model):
            observed.append((step, model.token_embedding.weight.detach().clone()))
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            first, history = run_training(DEMO_TRAIN, DEMO_VALIDATION, steps=4,
                output_dir=Path(directory) / 'observed', on_step=observer)
            second, repeated = run_training(DEMO_TRAIN, DEMO_VALIDATION, steps=4,
                output_dir=Path(directory) / 'plain')
        self.assertEqual([step for step, _ in observed], [1, 2, 3, 4])
        self.assertFalse(torch.equal(observed[0][1], observed[-1][1]))
        self.assertEqual(history, repeated)
        for key, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[key]))

    def test_repeatability_learning_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            first, history = run_training(DEMO_TRAIN, DEMO_VALIDATION, steps=100,
                                          output_dir=Path(directory) / "one")
            second, repeated = run_training(DEMO_TRAIN, DEMO_VALIDATION, steps=100,
                                            output_dir=Path(directory) / "two")
            self.assertEqual(history, repeated)
            for key, value in first.state_dict().items():
                self.assertTrue(torch.equal(value, second.state_dict()[key]))
            self.assertLess(history[-1]['train_loss'], history[0]['train_loss'])
            self.assertLess(history[-1]['validation_loss'], history[0]['validation_loss'])
            checkpoint = torch.load(Path(directory) / "one/checkpoint.pt", weights_only=True)
            restored = GPT(**checkpoint['model_config'])
            restored.load_state_dict(checkpoint['model_state_dict'])
            ids = torch.zeros(1, 3, dtype=torch.long)
            with torch.no_grad():
                torch.testing.assert_close(first(ids), restored(ids), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
