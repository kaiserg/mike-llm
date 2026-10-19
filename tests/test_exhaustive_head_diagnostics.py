import unittest
import torch
from gpt import GPT
from head_diagnostics import forward_head_masks
from exhaustive_head_diagnostics import forward_all_head_masks,blocked_pairs,analyze_mask_cube


class ExhaustiveTests(unittest.TestCase):
    def test_all_masks_match_full_window_reference(self):
        torch.set_num_threads(1)
        for encoding in ('learned','sinusoidal'):
            torch.manual_seed(21)
            model=GPT(9,64,positional_encoding=encoding).eval()
            before={k:v.clone() for k,v in model.state_dict().items()}
            for length in (1,8,12,16,32,64):
                ids=torch.randint(0,9,(2,length))
                cached=forward_all_head_masks(model,ids)
                self.assertEqual(cached.shape,(2,64,9))
                for mask in range(64):
                    reference=forward_head_masks(model,ids,blocked_pairs(mask))
                    torch.testing.assert_close(cached[:,mask],reference,rtol=1e-5,atol=1e-6)
            for key,value in model.state_dict().items():self.assertTrue(torch.equal(value,before[key]))

    def test_known_interaction_cube_and_efficiency(self):
        # Explicit gain: three individual effects, a pair and a three-way effect.
        gains=[]
        for mask in range(64):
            x=[bool(mask&(1<<i)) for i in range(6)]
            gains.append(1*x[0]+2*x[1]-3*x[2]+4*x[0]*x[1]-6*x[0]*x[1]*x[2])
        result=analyze_mask_cube([10-g for g in gains])
        coefficients=result['mobius_gain_coefficients']
        self.assertEqual(coefficients[3],4)
        self.assertEqual(coefficients[7],-6)
        for mask in range(64):
            recovered=sum(coefficients[s] for s in range(64) if s&mask==s)
            self.assertAlmostEqual(recovered,gains[mask])
        self.assertAlmostEqual(sum(result['shapley_gains'].values()),gains[63])
        self.assertAlmostEqual(result['shapley_gains']['block1_head1'],1+4/2-6/3)
        self.assertAlmostEqual(result['shapley_gains']['block1_head2'],2+4/2-6/3)
        self.assertEqual(result['conditional_edges']['block2_head1']['positive_contexts'],0)
        for bad in (-1,64,True):
            with self.assertRaises(ValueError):blocked_pairs(bad)
