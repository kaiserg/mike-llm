import contextlib
import io
import math
from pathlib import Path
import tempfile
import unittest

import torch

from generate import generate, load_checkpoint
from gpt import GPT
from position_diagnostics import score_positions
import torch.nn.functional as F
from sinusoidal_positions import SinusoidalPositionEmbedding
from train import DEMO_TRAIN, DEMO_VALIDATION, run_training


class SinusoidalTests(unittest.TestCase):
    def test_formula_even_odd_and_buffers(self):
        for width in (1, 5, 24):
            layer=SinusoidalPositionEmbedding(8,width)
            result=layer(torch.arange(8))
            for p in (0,1,7):
                for j in range(width):
                    angle=p/(10000**(2*(j//2)/width))
                    expected=math.sin(angle) if j%2==0 else math.cos(angle)
                    self.assertAlmostEqual(result[p,j].item(),expected,places=6)
            self.assertEqual(sum(p.numel() for p in layer.parameters()),0)
            self.assertFalse(layer.table.requires_grad)
            self.assertEqual(layer.to(dtype=torch.float64).table.dtype,torch.float64)

    def test_shared_initialization_and_causality(self):
        torch.manual_seed(7)
        short=GPT(vocab_size=4,context_length=16,embedding_dim=24,num_heads=3,positional_encoding='sinusoidal')
        torch.manual_seed(7)
        long=GPT(vocab_size=4,context_length=64,embedding_dim=24,num_heads=3,positional_encoding='sinusoidal')
        for (name,a),(other,b) in zip(short.named_parameters(),long.named_parameters()):
            self.assertEqual(name,other)
            self.assertTrue(torch.equal(a,b))
        x=torch.randint(0,4,(2,12))
        with torch.no_grad():
            torch.testing.assert_close(short(x),long(x),rtol=0,atol=0)
            changed=x.clone();changed[:,8:]=(changed[:,8:]+1)%4
            torch.testing.assert_close(long(changed)[:,:8],long(x)[:,:8])
        for invalid in ('rotary','invalid'):
            with self.assertRaises(ValueError):GPT(4,16,positional_encoding=invalid)

    def test_position_shift_diagnostic_supports_fixed_table(self):
        model=GPT(4,8,positional_encoding='sinusoidal')
        ids=torch.tensor([0,1,2,3,0,1,2,3,0,1])
        targets=[8,9]
        result=score_positions(model,ids,targets,[3],position_offset=5)[0]
        inputs=torch.stack([ids[t-3:t] for t in targets])
        with torch.no_grad():
            x=model.token_embedding(inputs)+model.position_embedding.table[5:8]
            logits=model.lm_head(model.final_norm(model.blocks(x)))[:,-1]
            expected=F.cross_entropy(logits.double(),ids[targets]).item()
        self.assertAlmostEqual(result['mean_nll'],expected,places=6)
        self.assertEqual(len(model.position_embedding._forward_hooks),0)

    def test_checkpoint_training_generation_and_legacy_default(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            model,_=run_training(DEMO_TRAIN,DEMO_VALIDATION,steps=4,
                positional_encoding='sinusoidal',output_dir=Path(tmp)/'sin')
            loaded,tok=load_checkpoint(Path(tmp)/'sin/checkpoint.pt')
            self.assertTrue(torch.equal(model.position_embedding.table,loaded.position_embedding.table))
            ids=torch.tensor([tok.encode('a model')])
            with torch.no_grad():torch.testing.assert_close(model(ids),loaded(ids),rtol=0,atol=0)
            self.assertEqual(generate(loaded,ids,3,0).shape[1],ids.shape[1]+3)
        torch.manual_seed(42);default=GPT(4,16)
        torch.manual_seed(42);explicit=GPT(4,16,positional_encoding='learned')
        for key,value in default.state_dict().items():self.assertTrue(torch.equal(value,explicit.state_dict()[key]))
        self.assertIn('position_embedding.weight',default.state_dict())


if __name__=='__main__':unittest.main()
