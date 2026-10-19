# Exhaustive six-head mask combinations

[Documentation index](README.md) · [Previous](29-joint-head-history-diagnostics.md) · [Next](31-stage-closure-and-test.md)

Driver: [`scripts/analyze_all_head_masks.py`](../scripts/analyze_all_head_masks.py). Cached evaluator and exact interaction analysis: [`src/exhaustive_head_diagnostics.py`](../src/exhaustive_head_diagnostics.py). [Full report](results/all-head-mask-diagnostics.json), with all masks, rankings, control checks and coefficients.

## Fixed experiment

Evaluate all **64 subsets** of six head restrictions in each of the same two frozen context-64 checkpoints: learned seed 7 and sinusoidal seed 123. Each subset scores the identical **135522 validation next-character targets**. There is no training and no test-text access.

The reference permits every head to read the final sixteen input tokens only. A mask bit restricts one head to the latest eight, in all queries, within the same outer sixteen-token isolation boundary. Keep original position IDs: in a full-sized window the retained sixteen occupy 48–63, and a restricted head's latest eight occupy 56–63. Early windows naturally have fewer tokens. This is the 8/16 fixed-position intervention from chapters 28–29, **not** a search over head removal or native 64-token attention.

Mask bit order, least significant first: **B1H1, B1H2, B1H3, B2H1, B2H2, B2H3**. Block/head labels start at one; bit numbering starts at zero. Mask 0 is baseline16, mask 63 is all8. All other mask IDs uniquely identify subsets without ambiguity.

## Efficient evaluation and numerical agreement

Suffix isolation makes older positions irrelevant to every retained representation. The evaluator therefore processes just the sixteen-token suffix while preserving its original positional IDs, computes both outputs (eight- and sixteen-token access) per head, reuses the first-block combinations, and caches second-block head outputs separately for each first-block state. This changes computation sharing, not the mask experiment. No weights, buffers or hooks are modified.

All 64 combinations are tested against the original full-window implementation for both encodings at input lengths 1, 8, 12, 16, 32 and 64 (rtol 1e-5, atol 1e-6). The driver independently checks all available archived individual/joint controls within 1e-6 NLL. Exact runtime differences are in the report; floating-point kernels can produce tiny differences when tensor shapes change. Results remain measured full-validation scores, with no target subsampling.

## How to read the interaction summaries

Let N(S) be validation NLL when the heads in S are restricted, and G(S)=N(empty)−N(S). Positive gain means lower prediction loss.

- **Conditional head gain:** G(S∪{h})−G(S), for all 32 subsets S of the other five heads. The range and positive-context count show whether a head's mask changes sign across contexts.
- **Shapley gain:** the exact weighted average of those marginal gains over all head orderings. It allocates G(all six)−G(empty) among six masks; it does not score only the winning subset or prove intrinsic head importance. Computation uses all subsets, not sampled permutations.
- **Pair interaction:** G(S∪{a,b})−G(S∪{a})−G(S∪{b})+G(S), evaluated over all 16 subsets of the other four heads. Positive values mean extra combined improvement beyond the two conditional gains in that context; sign can vary with S. The displayed mean weights the sixteen contexts equally.
- **Interaction expansion:** coefficients c(T) satisfy G(S)=sum of c(T) for T⊆S. Singleton coefficients are individual baseline gains; orders 2–6 capture the exact residual structure beyond them. Signed sums across all orders recover G(all six). Absolute sums describe coefficient magnitudes and cancellation, not independently attributable predictive importance.

## Learned seed 7

### Main comparisons

| Reference | Mask ID | Blocked added-history access | NLL | PPL |
| --- | --- | --- | --- | --- |
| baseline16 | 0 | none | 2.268860 | 9.668370 |
| previous selected mask | 57 | B1H1, B2H1, B2H2, B2H3 | 2.228418 | 9.285167 |
| all8 | 63 | B1H1, B1H2, B1H3, B2H1, B2H2, B2H3 | 2.225004 | 9.253523 |
| best of 64 | 63 | B1H1, B1H2, B1H3, B2H1, B2H2, B2H3 | 2.225004 | 9.253523 |

### Best eight combinations

| Rank | Mask ID | Blocked heads | NLL | PPL | NLL above best |
| --- | --- | --- | --- | --- | --- |
| 1 | 63 | B1H1, B1H2, B1H3, B2H1, B2H2, B2H3 | 2.225004 | 9.253523 | +0.000000 |
| 2 | 47 | B1H1, B1H2, B1H3, B2H1, B2H3 | 2.226223 | 9.264809 | +0.001219 |
| 3 | 31 | B1H1, B1H2, B1H3, B2H1, B2H2 | 2.226849 | 9.270606 | +0.001844 |
| 4 | 15 | B1H1, B1H2, B1H3, B2H1 | 2.227965 | 9.280958 | +0.002960 |
| 5 | 59 | B1H1, B1H2, B2H1, B2H2, B2H3 | 2.228028 | 9.281544 | +0.003024 |
| 6 | 57 | B1H1, B2H1, B2H2, B2H3 | 2.228418 | 9.285167 | +0.003414 |
| 7 | 43 | B1H1, B1H2, B2H1, B2H3 | 2.229194 | 9.292374 | +0.004190 |
| 8 | 61 | B1H1, B1H3, B2H1, B2H2, B2H3 | 2.229408 | 9.294365 | +0.004404 |

### Best mask at each number of restricted heads

| Restricted heads | Mask ID | Blocked heads | NLL |
| --- | --- | --- | --- |
| 0 | 0 | none | 2.268860 |
| 1 | 1 | B1H1 | 2.241213 |
| 2 | 9 | B1H1, B2H1 | 2.231587 |
| 3 | 41 | B1H1, B2H1, B2H3 | 2.229769 |
| 4 | 15 | B1H1, B1H2, B1H3, B2H1 | 2.227965 |
| 5 | 47 | B1H1, B1H2, B1H3, B2H1, B2H3 | 2.226223 |
| 6 | 63 | B1H1, B1H2, B1H3, B2H1, B2H2, B2H3 | 2.225004 |

### All 64 validation losses

Rows encode the three block-1 bits (B1H1 is bit 0); columns encode the three block-2 bits (B2H1 is bit 0 within the block). Full mask ID = row + 8×column. A one bit restricts that head to eight tokens. Values are NLL.

| B1 bits / B2 bits | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.268860 | 2.259034 | 2.264380 | 2.254884 | 2.266393 | 2.256676 | 2.261795 | 2.252403 |
| 1 | 2.241213 | 2.231587 | 2.239672 | 2.230334 | 2.239278 | 2.229769 | 2.237650 | 2.228418 |
| 2 | 2.268870 | 2.258836 | 2.264522 | 2.254838 | 2.266412 | 2.256489 | 2.261945 | 2.252368 |
| 3 | 2.240872 | 2.231023 | 2.239499 | 2.229955 | 2.238913 | 2.229194 | 2.237454 | 2.228028 |
| 4 | 2.270828 | 2.260874 | 2.266631 | 2.257030 | 2.268599 | 2.258744 | 2.264291 | 2.254785 |
| 5 | 2.242113 | 2.232193 | 2.240813 | 2.231205 | 2.240284 | 2.230488 | 2.238905 | 2.229408 |
| 6 | 2.268545 | 2.258081 | 2.264160 | 2.254065 | 2.266276 | 2.255928 | 2.261773 | 2.251791 |
| 7 | 2.238179 | 2.227965 | 2.236736 | 2.226849 | 2.236290 | 2.226223 | 2.234756 | 2.225004 |

### Conditional head effects and exact Shapley allocation

| Head | Individual gain | Minimum conditional gain | Maximum conditional gain | Contexts helped | Shapley gain |
| --- | --- | --- | --- | --- | --- |
| block1_head1 | +0.027647 | +0.023985 | +0.030366 | 32/32 | +0.027006 |
| block1_head2 | -0.000011 | -0.000150 | +0.004404 | 28/32 | +0.001934 |
| block1_head3 | -0.001969 | -0.002496 | +0.003107 | 16/32 | +0.000212 |
| block2_head1 | +0.009825 | +0.009232 | +0.010464 | 32/32 | +0.009772 |
| block2_head2 | +0.004480 | +0.000988 | +0.004598 | 32/32 | +0.002797 |
| block2_head3 | +0.002467 | +0.001705 | +0.002585 | 32/32 | +0.002133 |

### Largest mean pair interactions

| Pair | Mean interaction | Minimum | Maximum | Positive contexts |
| --- | --- | --- | --- | --- |
| B1H2 + B1H3 | +0.003215 | +0.002294 | +0.004014 | 16/16 |
| B1H1 + B2H2 | -0.002930 | -0.003008 | -0.002856 | 0/16 |
| B1H1 + B1H3 | +0.001822 | +0.001068 | +0.002526 | 16/16 |
| B1H1 + B1H2 | +0.000943 | +0.000315 | +0.001672 | 16/16 |
| B1H1 + B2H3 | -0.000477 | -0.000565 | -0.000380 | 0/16 |

### Exact interaction expansion by order

| Order | Coefficients | Signed sum | Sum of absolute coefficients |
| --- | --- | --- | --- |
| 1 | 6 | +0.042440 | 0.046399 |
| 2 | 15 | -0.000599 | 0.008938 |
| 3 | 20 | +0.002276 | 0.002539 |
| 4 | 15 | -0.000266 | 0.000311 |
| 5 | 6 | +0.000003 | 0.000012 |
| 6 | 1 | +0.000001 | 0.000001 |


## Sinusoidal seed 123

### Main comparisons

| Reference | Mask ID | Blocked added-history access | NLL | PPL |
| --- | --- | --- | --- | --- |
| baseline16 | 0 | none | 2.098424 | 8.153309 |
| previous selected mask | 61 | B1H1, B1H3, B2H1, B2H2, B2H3 | 2.016729 | 7.513707 |
| all8 | 63 | B1H1, B1H2, B1H3, B2H1, B2H2, B2H3 | 2.013815 | 7.491846 |
| best of 64 | 63 | B1H1, B1H2, B1H3, B2H1, B2H2, B2H3 | 2.013815 | 7.491846 |

### Best eight combinations

| Rank | Mask ID | Blocked heads | NLL | PPL | NLL above best |
| --- | --- | --- | --- | --- | --- |
| 1 | 63 | B1H1, B1H2, B1H3, B2H1, B2H2, B2H3 | 2.013815 | 7.491846 | +0.000000 |
| 2 | 61 | B1H1, B1H3, B2H1, B2H2, B2H3 | 2.016729 | 7.513707 | +0.002914 |
| 3 | 62 | B1H2, B1H3, B2H1, B2H2, B2H3 | 2.017347 | 7.518350 | +0.003532 |
| 4 | 60 | B1H3, B2H1, B2H2, B2H3 | 2.018635 | 7.528041 | +0.004820 |
| 5 | 59 | B1H1, B1H2, B2H1, B2H2, B2H3 | 2.024666 | 7.573580 | +0.010851 |
| 6 | 57 | B1H1, B2H1, B2H2, B2H3 | 2.030307 | 7.616425 | +0.016492 |
| 7 | 47 | B1H1, B1H2, B1H3, B2H1, B2H3 | 2.032986 | 7.636852 | +0.019170 |
| 8 | 58 | B1H2, B2H1, B2H2, B2H3 | 2.034504 | 7.648460 | +0.020689 |

### Best mask at each number of restricted heads

| Restricted heads | Mask ID | Blocked heads | NLL |
| --- | --- | --- | --- |
| 0 | 0 | none | 2.098424 |
| 1 | 8 | B2H1 | 2.077448 |
| 2 | 40 | B2H1, B2H3 | 2.056179 |
| 3 | 44 | B1H3, B2H1, B2H3 | 2.038472 |
| 4 | 60 | B1H3, B2H1, B2H2, B2H3 | 2.018635 |
| 5 | 61 | B1H1, B1H3, B2H1, B2H2, B2H3 | 2.016729 |
| 6 | 63 | B1H1, B1H2, B1H3, B2H1, B2H2, B2H3 | 2.013815 |

### All 64 validation losses

Rows encode the three block-1 bits (B1H1 is bit 0); columns encode the three block-2 bits (B2H1 is bit 0 within the block). Full mask ID = row + 8×column. A one bit restricts that head to eight tokens. Values are NLL.

| B1 bits / B2 bits | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.098424 | 2.077448 | 2.078194 | 2.059172 | 2.077878 | 2.056179 | 2.058870 | 2.039507 |
| 1 | 2.094138 | 2.069973 | 2.072023 | 2.049896 | 2.073302 | 2.048435 | 2.052556 | 2.030307 |
| 2 | 2.098997 | 2.074444 | 2.076152 | 2.053735 | 2.078069 | 2.052980 | 2.057009 | 2.034504 |
| 3 | 2.093346 | 2.065174 | 2.070419 | 2.044566 | 2.071583 | 2.042886 | 2.050453 | 2.024666 |
| 4 | 2.091871 | 2.062084 | 2.067073 | 2.040017 | 2.069142 | 2.038472 | 2.045924 | 2.018635 |
| 5 | 2.088352 | 2.059535 | 2.064420 | 2.037873 | 2.065964 | 2.036245 | 2.043526 | 2.016729 |
| 6 | 2.092646 | 2.060885 | 2.067847 | 2.038672 | 2.069513 | 2.036978 | 2.046642 | 2.017347 |
| 7 | 2.088257 | 2.056804 | 2.064057 | 2.035081 | 2.065265 | 2.032986 | 2.042922 | 2.013815 |

### Conditional head effects and exact Shapley allocation

| Head | Individual gain | Minimum conditional gain | Maximum conditional gain | Contexts helped | Shapley gain |
| --- | --- | --- | --- | --- | --- |
| block1_head1 | +0.004286 | +0.001906 | +0.010095 | 32/32 | +0.004843 |
| block1_head2 | -0.000573 | -0.000775 | +0.005641 | 26/32 | +0.001717 |
| block1_head3 | +0.006553 | +0.005089 | +0.020872 | 32/32 | +0.010189 |
| block2_head1 | +0.020976 | +0.019022 | +0.032535 | 32/32 | +0.026032 |
| block2_head2 | +0.020230 | +0.016672 | +0.024799 | 32/32 | +0.020645 |
| block2_head3 | +0.020546 | +0.019143 | +0.023906 | 32/32 | +0.021183 |

### Largest mean pair interactions

| Pair | Mean interaction | Minimum | Maximum | Positive contexts |
| --- | --- | --- | --- | --- |
| B1H3 + B2H1 | +0.005858 | +0.003124 | +0.008970 | 16/16 |
| B1H1 + B1H3 | -0.004120 | -0.007294 | -0.000766 | 0/16 |
| B1H2 + B2H1 | +0.002906 | +0.001865 | +0.004007 | 16/16 |
| B2H1 + B2H2 | -0.002605 | -0.003381 | -0.001954 | 0/16 |
| B1H3 + B2H2 | +0.002075 | +0.000950 | +0.004568 | 16/16 |

### Exact interaction expansion by order

| Order | Coefficients | Signed sum | Sum of absolute coefficients |
| --- | --- | --- | --- |
| 1 | 6 | +0.072017 | 0.073163 |
| 2 | 15 | +0.025444 | 0.033734 |
| 3 | 20 | -0.015600 | 0.017912 |
| 4 | 15 | +0.002904 | 0.004862 |
| 5 | 6 | -0.000059 | 0.000742 |
| 6 | 1 | -0.000098 | 0.000098 |

## Interpretation

**Mask 63—all six heads restricted to the latest eight—is the unique lowest-NLL combination in both checkpoints.** It is also the only single-bit-flip local minimum in each measured cube. This establishes the best mask within the complete 64-subset grid, rather than only within the previously selected subset of masks.

- **Learned seed 7:** baseline16 PPL is **9.668370**, the previous selected-mask PPL is **9.285167**, and the exhaustive winner scores **9.253523** (NLL **2.225004**). The runner-up is mask **47**, which leaves only B2H2 with sixteen-token access; its NLL is higher by **0.001219** and PPL is **9.264809**.
- **Sinusoidal seed 123:** baseline16 PPL is **8.153309**, the previous selected mask scores **7.513707**, and mask 63 scores **7.491846** (NLL **2.013815**). The runner-up is mask **61**, the prior selected mask, which leaves only B1H2 with sixteen-token access; its NLL is higher by **0.002914**.

The maximum absolute differences from archived control losses are approximately **1.06e-8** learned and **1.54e-8** sinusoidal, much smaller than either winner-versus-runner-up gap. No other mask is within 1e-4 NLL of the winner. This addresses numerical agreement, not statistical significance or independent generalization.

**Some masks help consistently; others change sign with their companions.** Learned B1H1 and all three block-2 head masks improve loss in all 32 possible contexts of the other heads. Learned B1H2 improves in 28/32 and B1H3 in 16/32, despite both having negative individual baseline gains. Sinusoidal B1H2 changes sign (26/32 positive contexts); the other five masks help in all 32. Thus the earlier individual “beneficial access” labels do not transfer to every coalition. All six Shapley gains are positive in both cubes, but that global allocation does not mean every individual intervention is beneficial in every context.

**The dominant allocations retain the earlier block distinction.** Learned B1H1 receives the largest exact Shapley gain (**0.027006**), followed by B2H1 (**0.009772**). In the sinusoidal checkpoint, the three block-2 allocations dominate (**0.026032**, **0.020645**, **0.021183**), with a substantial smaller contribution from B1H3 (**0.010189**). These allocations refer to the entire baseline16→all8 coalition game, not just the previously selected heads.

**Pair interactions have stable and opposing signs.** In the learned cube, B1H2+B1H3 has positive interaction in every background context (mean **+0.003215**), while B1H1+B2H2 is negative throughout (mean **−0.002930**). In the sinusoidal cube, B1H3+B2H1 is positive throughout (mean **+0.005858**), whereas B1H1+B1H3 is negative throughout (mean **−0.004120**). These net mask interactions do not establish direct communication between specific heads.

**Pairs alone do not reconstruct the cube.** For the learned checkpoint, the signed sums are approximately +0.042440 from individual terms, −0.000599 from pairs, and +0.002276 from triples, with smaller higher-order corrections. For sinusoidal positions, individuals contribute +0.072017, pairs +0.025444 and triples −0.015600, again with higher-order corrections. The substantial cancellation explains why summing the individual effects, or even adding only pair effects, is insufficient. Exact coefficients and all conditional ranges are archived.

The experiment identifies a finite-grid optimum and measures its interactions; it does not show that eight tokens are generally sufficient or preferable. Both checkpoints were selected because of their specific 8→16 reversals. Native 64-token history is outside this mask grid and remains a separate comparison. Neither the global optimum here nor a positive Shapley allocation is evidence that a head should be permanently pruned.

The winning masks are selected on the same validation book used to measure them. They are the best within this finite intervention grid for these two checkpoints, not independently validated model improvements. Repeated validation selection can overfit the book, and tiny score gaps do not establish significance. This exhaustive experiment closes the six-bit subset search under the frozen protocol; it does not generalize the result to other training seeds, encodings in general, native long history or new corpora. The test book stays reserved.

## Reproduce and verify

```bash
source .venv/bin/activate
python scripts/analyze_all_head_masks.py --output runs/all-head-masks/report.json
PYTHONPATH=src python -m unittest discover -s tests -v
```

Use a fresh output path for a repeat. The driver verifies checkpoint/validation hashes, matches archived controls, and records base commit, source hashes identifying initially uncommitted code, input archive hashes, CPU and runtime versions, deterministic operations, one thread and batch size 128. All 128 checkpoint/mask scores contain 135522 targets each. No checkpoint is rewritten or production architecture changed.

All 40 tests pass. The added tests cover full-mask equivalence with the reference and unchanged model state, plus a known synthetic cube with individual, pair and third-order terms to verify exact reconstruction and Shapley efficiency. Existing tests cover per-head and joint masks, original-ID suffix isolation, and short-window behavior.
