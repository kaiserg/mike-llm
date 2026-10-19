# Positional loss diagnostics: learned versus sinusoidal

[Documentation index](README.md) · [Previous](23-sinusoidal-positions.md)

Reproduction driver: [`scripts/compare_position_diagnostics.py`](../scripts/compare_position_diagnostics.py). Full [sinusoidal prefix/shift results](results/sinusoidal-position-diagnostics.json), [sinusoidal complete-window results](results/sinusoidal-window-position-diagnostics.json), and [paired encoding comparison](results/positional-encoding-diagnostic-comparison.json). Learned reference arrays are in [chapter 21's prefix report](results/position-diagnostics.json) and [window report](results/window-position-diagnostics.json).

## Answer

Replacing learned positions with sinusoidal ones does **not** establish more reliable use of long history. On identical targets, the sinusoidal context-64 model's mean NLL rises from **1.915872** with 16 history characters to **1.948275** with 64. Three seeds worsen and one improves slightly. The corresponding learned-position increase is **1.944632 → 1.968524**, with all four seeds worsening.

The sinusoidal context-64 checkpoint family has lower absolute loss than the learned counterpart at many intermediate/late prefix lengths. That is a better prediction score for that configured model, not the same as benefiting from the extra 48 characters. The additional-history penalty is actually larger on average in the sinusoidal diagnostic, though also much more variable across seeds.

## Same checkpoints, same targets, no training

All models are the frozen 10000-update configurations from chapters 20 and 23: four training seeds, the same 1280000-target budget per model, unchanged corpus, and native capacities 16 or 64. This study does not use the 40000-update checkpoints from chapter 22 as encoding controls. It retrains no model and never reads test text.

We re-executed both encoding families with the current same diagnostic code. The learned prefix losses, offset losses, and complete-window losses reproduced their archived numerical arrays exactly. Checkpoint and validation hashes are verified before scoring. The reports identify source versions and all input artifacts.

The controlled-prefix analysis scores **2117 identical target characters**, at validation indices 64, 128, and so on. Each target has at least 64 preceding characters. Prefix length L uses its immediate L preceding tokens and native position IDs `0..L-1`; only the final logits are scored. All lengths, encodings, contexts, and seeds use the same target pool. These subset losses should not be substituted for the full-book scores in chapter 23.

The complete-window analysis instead scores **135488** targets in 2117 aligned, nonoverlapping 64-input blocks, omitting the final 34 possible targets to keep blocks complete. Context 16 resets inside each of the four 16-token chunks, while context 64 sees one whole block. Thus column-wise model comparisons have identical targets, but their available histories differ after chunk boundaries.

## Reproduce

From the repository root with the environment activated and study checkpoints retained:

```bash
python scripts/compare_position_diagnostics.py
```

The driver runs or verifies cached prefix and window analyses for both encodings under `runs/sinusoidal-position-diagnostics/`, checks identical targets/data, compares learned outputs to their archives, and writes `comparison.json`. An existing final comparison is not overwritten; to reproduce into a fresh directory:

```bash
python scripts/compare_position_diagnostics.py --output-dir runs/sinusoidal-position-repeat
```

The underlying generic analyzers now record sinusoidal-module provenance as well as GPT/helper provenance. Complete-window reports also retain checkpoint paths and hashes. Their learned defaults and numerical scoring behavior are unchanged.

## 1. Loss by native position and available history

Mean NLL across four training seeds; the final zero-based position ID is L-1:

| History L | Learned 16 | Sinusoidal 16 | Learned 64 | Sinusoidal 64 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2.462595 | 2.484142 | 2.531668 | 2.557216 |
| 2 | 2.115496 | 2.137622 | 2.180070 | 2.215944 |
| 4 | 1.903431 | 1.926278 | 1.968737 | 1.930084 |
| 8 | 1.885773 | 1.903566 | 1.945018 | 1.910277 |
| 12 | 1.889051 | 1.907261 | 1.933253 | 1.910013 |
| 16 | 1.906228 | 1.926774 | 1.944632 | 1.915872 |
| 24 | — | — | 1.952733 | 1.917705 |
| 32 | — | — | 1.943035 | 1.912420 |
| 40 | — | — | 1.954315 | 1.924605 |
| 48 | — | — | 1.958933 | 1.925543 |
| 56 | — | — | 1.962617 | 1.917448 |
| 64 | — | — | 1.968524 | 1.948275 |

Both encodings gain most from the first few characters. The sinusoidal context-64 model improves relative to its learned counterpart at history lengths 4 and above in this selected table, but extending its own history from 16 to 64 is not beneficial on average. At lengths 1–2 it still has higher loss than the learned context-64 model. There is no uniform improvement at every position.

Within one native prefix curve, increasing L changes both the available token history and the final absolute position ID. It therefore does not isolate memory from positional effects. Likewise, comparing encodings changes trained weights, parameter counts, positional signal scale, and initialization details as described in chapter 23. The curves identify predictive behavior, not a unique training cause.

### Per-seed effect of extending context64 history from 16 to 64

Positive differences mean the longer prefix gives higher loss on the same targets:

| Seed | Learned ΔNLL | Sinusoidal ΔNLL |
| --- | ---: | ---: |
| 42 | +0.020745 | +0.012130 |
| 7 | +0.031705 | -0.003180 |
| 123 | +0.012873 | +0.079946 |
| 2026 | +0.030243 | +0.040715 |

Learned mean ΔNLL is **+0.023891 ± 0.008807**; sinusoidal mean is **+0.032403 ± 0.036545** (sample SD over seeds). Sinusoidal seed 7 gains only about 0.00318 nats per target from longer history, while seed 123 worsens substantially. This is not evidence of a stable long-history gain across seeds. It does not mean every target fails to benefit, nor that the model literally ignores distant tokens.

A matched-target decomposition clarifies why a narrower context-model gap need not imply useful distant history:

$$L_{64}(64)-L_{16}(16)=[L_{64}(16)-L_{16}(16)]+[L_{64}(64)-L_{64}(16)]$$

| Encoding | Same 16-token-history model gap | Extra-history change in context64 | Native model gap |
| --- | ---: | ---: | ---: |
| Learned | +0.038405 | +0.023891 | +0.062296 |
| Sinusoidal | -0.010902 | +0.032403 | +0.021500 |

The sinusoidal long-trained model scores slightly better than the sinusoidal short-trained model when both receive the same 16-token history on this subset. Its extra 48 characters then add a loss penalty. Thus the smaller native model gap comes from the common-history comparison, not a positive additional-history effect. This arithmetic decomposition is descriptive, not a causal attribution.

## 2. Fixed history, shifted absolute IDs

We keep exactly the same 16 input characters and target characters, and shift every position ID inside the context-64 model. The final IDs are 15, 31, 47, and 63. No weights change.

| Position IDs | Learned mean NLL | Sinusoidal mean NLL |
| --- | ---: | ---: |
| 0–15 | 1.944632 | 1.915872 |
| 16–31 | 1.956883 | 1.926185 |
| 32–47 | 2.019795 | 1.964658 |
| 48–63 | 2.052882 | 1.986378 |

Sinusoidal functions are fixed, but adding their absolute vectors to embeddings does not make the trained Transformer invariant to a common position shift. Both encodings show increased mean loss for late IDs in this intervention.

The penalty for shifting IDs 0–15 to 48–63 is:

| Seed | Learned ΔNLL | Sinusoidal ΔNLL |
| --- | ---: | ---: |
| 42 | +0.021954 | +0.031853 |
| 7 | +0.353079 | +0.033933 |
| 123 | +0.027229 | +0.168367 |
| 2026 | +0.030737 | +0.047872 |

The mean penalty falls from **+0.108250** learned to **+0.070506** sinusoidal, but this does not mean every seed becomes less sensitive. Sinusoidal penalties are larger in three of four paired seeds; the lower overall mean is strongly affected by removing the unusually large learned penalty in seed 7. The sinusoidal outlier is instead seed 123. Across-seed variation matters more than reporting only the aggregate reduction.

A short prefix placed at IDs 48–63 has no tokens at IDs 0–47, unlike a native full training window. The intervention changes all sixteen positional vectors and their interaction with attention, not just one final-position parameter. It is therefore a distribution-changing positional sensitivity test, not proof of a positional-table defect or an isolated measure of long-context utility.

## 3. Complete-window loss can reverse the apparent winner

Grouped column means on the same aligned blocks:

| Block columns | Learned 16 | Sinusoidal 16 | Learned 64 | Sinusoidal 64 |
| --- | ---: | ---: | ---: | ---: |
| 0–15 | 1.936525 | 1.961723 | 1.998374 | 1.976347 |
| 16–31 | 1.936176 | 1.962964 | 1.945977 | 1.911753 |
| 32–47 | 1.945415 | 1.968882 | 1.962387 | 1.922200 |
| 48–63 | 1.937554 | 1.957213 | 1.947368 | 1.911897 |
| All columns | 1.938918 | 1.962696 | 1.963526 | 1.930549 |

The sinusoidal context-64 model wins the all-position window average over sinusoidal context 16 (**1.930549 versus 1.962696**), even though its native rolling/full-book loss is worse in chapter 23. This is a protocol-dependent ordering, not a contradiction.

Context 16 restarts history every sixteen inputs in this window experiment, repeatedly exposing difficult one-, two-, and three-character prefixes. Context 64 restarts only once. The later-column advantage can therefore arise without needing information more than sixteen characters back. Different block columns also predict different character pools. The controlled-prefix comparison, which fixes targets and explicitly compares histories 16 and 64, remains the relevant check for the additional-history question.

The last column is a cross-check: its target indices are exactly the controlled-prefix target indices. Context16's last column matches its length-16 curve endpoint, while context64's matches length 64, within numerical tolerance. Full per-column arrays and every seed are preserved in the reports; no favorable columns were selected to determine a winner.

## Interpretation and limitations

The fixed representation changes the loss profile and improves the context-64 checkpoint family's absolute loss over much of the curve. It does not turn extending history past sixteen characters into a stable benefit. Lower positional-shift penalty in the mean is also not a uniform robustness gain, and lower complete-window average is not evidence that the extra 48 characters are necessary.

These results apply to the small character model, four seeds, fixed optimizer/budget, and this prepared literary validation text. The systematic stride-64 target subset could have sampling effects. Sample SD is descriptive seed variation, not a confidence interval or significance claim. Native curves still couple history with position IDs; shifted-prefix interventions have a distribution mismatch; comparisons of trained encodings have initialization/parameterization differences. No causal diagnosis of attention behavior or proof of long-range comprehension is established.

A more direct next ablation could keep the same last sixteen tokens and absolute position IDs while masking or replacing only older context, with clear distribution-shift caveats, or change the training/evaluation position mixture under a frozen protocol. The test book remains reserved.

## Verification

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

All 29 tests pass, including direct computation of sinusoidal position shifts and cleanup of temporary hooks. Both encoding studies use identical target indices, block starts, and validation hashes. Recomputed learned numerical results match their archives exactly. Source/checkpoint hashes, per-seed decompositions, prefix/window endpoint agreement, documentation examples, and links were checked. No training, new generation sampling, or test-text access was required.
