# Stage closure: validation selection and held-out test

[Documentation index](README.md) · [Previous](30-exhaustive-head-mask-diagnostics.md) · [Next](32-next-stage-plan.md)

Reproducible two-phase driver: [`scripts/finalize_stage.py`](../scripts/finalize_stage.py). [Frozen validation-only selection](results/final-model-selection.json). [Final held-out test result](results/final-test-evaluation.json).

## Outcome and project status

This stage is complete: Mike-LLM has a working small character-level GPT, reproducible training and generation, corpus provenance and disjoint book-level splits, checkpoint evaluation, and controlled experiments on training duration, context, positional encodings and attention masks. **It is a next-character language model, not an instruction-following or question-answering assistant.**

The selected model uses learned absolute positions, a training capacity of 64, seed 42 and 40000 updates. Its final **inference/evaluation window is 16 characters**, with IDs reset to 0–15. All heads use ordinary causal attention; there are **no per-head diagnostic masks** in the final configuration. This cropped/reset policy is different from the fixed-ID suffix-isolation experiments in chapters 25–30.

## Selection before test access

The objective was minimum already-measured **full-validation mean NLL**, without a training-budget constraint. Ties use the candidate ID. The selection inventory contains **35 existing operating configurations**, spanning learned/sinusoidal positions, training capacities 16/64, 2000/10000/40000 updates, four seeds where available, and native or previously evaluated capped windows. Duplicate earlier reports of the same 10000-step models are represented once in that inventory.

A separate audit scanned **341 archived scalar full-validation metrics**, including all diagnostic masks. No diagnostic score beat the chosen candidate. Target-subset position curves were excluded from that audit because they do not score the complete validation stream. The exact candidate inventory, validation scores, archive hashes and audit minimum are recorded in the selection file.

The chosen validation result was recomputed and matched its archive within 1e-10 NLL. The script then wrote the selection with exclusive creation and a UTC timestamp **before reading test content**. The test phase loaded that file, checked its hash, selected checkpoint, source hashes, preparation metadata and the validation minimum, and evaluated only that model/policy. Its report links the selection hash and both timestamps. No model or policy was changed after seeing test results.

The choice is a validation-selected checkpoint and operating policy, **not** a budget-matched architecture ranking or a statistically proven best seed. Choosing the best observed seed after substantial validation exploration can overfit Alice. The frozen test evaluation provides one additional book-level measurement rather than a new tuning signal.

## Validation family summary

Mean PPL is the arithmetic mean of the available seed perplexities; “best” is the lowest single-seed value. The table is ordered by family mean NLL. Single-seed early experiments are not estimates of seed variability. Native and cropped inference are separate operating configurations.

| Positions | Updates | Training capacity | Inference window | Seeds | Mean validation PPL | Best validation PPL |
| --- | --- | --- | --- | --- | --- | --- |
| learned | 40000 | 64 | 16 | 4 | 6.461221 | 6.288657 |
| learned | 40000 | 64 | 64 | 4 | 6.562562 | 6.438006 |
| learned | 10000 | 16 | 16 | 4 | 6.633624 | 6.572946 |
| sinusoidal | 10000 | 64 | 16 | 4 | 6.770595 | 6.643700 |
| sinusoidal | 10000 | 16 | 16 | 4 | 6.836397 | 6.734864 |
| learned | 10000 | 64 | 16 | 4 | 6.982041 | 6.869765 |
| sinusoidal | 10000 | 64 | 64 | 4 | 6.998999 | 6.756926 |
| learned | 10000 | 64 | 64 | 4 | 7.119825 | 6.948605 |
| learned | 2000 | 16 | 16 | 1 | 8.060321 | 8.060321 |
| learned | 2000 | 64 | 16 | 1 | 10.306627 | 10.306627 |
| learned | 2000 | 64 | 64 | 1 | 11.103687 | 11.103687 |

The selected 40000-update model used **5120000 training targets**, versus 1280000 at 10000 updates. It has no context-16/40000 counterpart in this study, so the final result does not prove that training capacity 64 is necessary or more efficient. It uses four times the training targets of the matched-budget experiments.

## Final configuration and provenance

| Field | Final value |
| --- | --- |
| Checkpoint | `runs/position-exposure-40000/seed-42/primary/checkpoint.pt` |
| Checkpoint SHA-256 | `1124b0352933d49816c057325ec58d891bec6aafaac51944c76b2ccae9777618` |
| Parameters | 17232 |
| Vocabulary | 28 character tokens: lowercase English letters, space and newline |
| Embedding width | 24 |
| Blocks / heads per block | 2 / 3 |
| FFN expansion | 4, with GELU |
| Normalization | Pre-LayerNorm and final LayerNorm |
| Positions | Learned absolute table, capacity 64 |
| Final inference policy | Last 16 characters; positions reset per window; native causal attention |
| Training | 40000 updates, seed 42, batch size 2, AdamW learning rate 0.003, weight decay 0.01, gradient clipping 1.0 |
| Training text | Prepared Frankenstein, 408305 characters |
| Validation text | Prepared Alice in Wonderland, 135523 characters |
| Test text | Prepared Pride and Prejudice, 694305 characters |

Checkpoint/tokenizer/model/training metadata are retained in the existing local checkpoint. The selected run's prior training was repeated with identical weights and histories (chapter 22); the closure does not retrain. Checkpoints and corpus files remain ignored local artifacts, so the Git documentation alone does not distribute a ready-to-load model. Existing corpus preparation and training drivers reproduce those prerequisites; the selection file records their expected hashes.

## Held-out results

| Split | Scored targets | NLL (nats/character) | Character PPL | Bits/character | Next-character accuracy |
| --- | --- | --- | --- | --- | --- |
| Validation: Alice in Wonderland | 135522 | 1.838747 | 6.288657 | 2.652752 | 44.85% |
| Test: Pride and Prejudice | 694304 | 1.749336 | 5.750783 | 2.523758 | 46.33% |

Test minus validation NLL is **-0.089411**. The first character of each split is not scored: every subsequent target uses only preceding characters from that same split, up to sixteen. No training/validation context is prepended to test. Both metrics use deterministic CPU evaluation with one thread and batch size 128, without temperature sampling. Lower PPL/loss is better; higher top-1 accuracy is better. Character PPL is not comparable directly with word- or subword-token perplexity.

The selected model scores lower loss on this test book than on the validation book, and its next-character accuracy rises from **44.85% to 46.33%**. Different books have different character/pattern distributions and predictability, so this should be read as one held-out measurement on Pride and Prejudice. It is not evidence that validation selection had no overfitting, that the model understands the book, or that the same score will hold on broader English text. No alternate checkpoint was evaluated on test to find a more favorable result.

**Test status has changed:** the Pride and Prejudice split has now been used for this final evaluation and is no longer untouched for future model development. Historical chapters accurately describe it as reserved at their time; future experiments must not treat repeated evaluation on this book as a fresh held-out confirmation.

## Consolidated experimental conclusions

1. **Training and evaluation work reproducibly on this small setup.** Checkpoint loading, causal behavior, shifted targets and CPU repeatability have executable checks. This is evidence about the implemented pipeline, not a guarantee of reproducibility across devices or library versions.
2. **More training helped, with a real budget cost.** Context-64/40000 checkpoints improved over their 10000-step versions; they processed four times the targets. Matching per-position exposure did not isolate architecture from total optimization budget.
3. **Longer training capacity did not guarantee useful longer inference.** Learned context 16 won all four native matched-10000-step pairs, while all four learned context-64/40000 checkpoints improved further with the cropped/reset 16-token evaluation. The final policy follows the measured validation result rather than assuming a longer window must be better.
4. **Sinusoidal positions changed profiles without a uniform improvement.** They did not consistently beat learned positions or establish dependable longer-context benefits. Numeric seed matching across parameterizations does not imply identical trainable initialization.
5. **Cropping/resetting positions and removing distant content are different interventions.** Fixed-ID tests found useful history beyond sixteen, including consistent 16→32 and 32→48 gains across the eight examined 10000-step checkpoints. This can coexist with cropped/reset inference scoring better.
6. **Attention effects are conditional and nonlinear.** The two selected 8→16 outliers localized differently across blocks and heads. Joint effects were not sums of individual gains. The exhaustive six-bit search found all-eight best in both selected outliers, with sign-changing conditional effects and higher-order interactions. Those are checkpoint-specific intervention results, not a universal head-pruning rule.
7. **The closure selects a simple inference policy rather than promoting diagnostic masks.** The winner is the existing longer-trained checkpoint with a cropped/reset window, and scores below every archived full-validation diagnostic mask under the selection audit.

## Capabilities and limits

The model can generate character-level continuations with learned local English spelling and literary-text patterns, with limited coherence. It was trained from scratch on one normalized novel and has no instruction tuning, conversational roles, question-answer pairs, retrieval, factual verification or tool use. It cannot be presented as a reliable assistant, general knowledge system, Spanish model or factual source about the books. The character normalization excludes uppercase and ordinary punctuation, and unknown characters cannot be encoded directly.

Each split contains one book. Splits are disjoint by document/source group, but this provides limited coverage of authors, genres and language. A single selected-checkpoint test score gives no test seed uncertainty or broad generalization estimate. The minimum over many validation experiments is selection-biased. Loss and next-character accuracy do not establish passage-level coherence, factuality, reasoning, memorization behavior or long-range understanding. Public-domain provenance and normalization are documented in chapter 14; no new legal or licensing assessment is implied here.

This closes the small-corpus educational experiment. A future stage should define a new question and evaluation plan before additional selection, broaden the licensed corpus and retain new untouched evaluation data. Instruction following would require a separate training objective/data stage after improving the base model, not just a generation setting.

## Reproduction and checks

With the existing prepared corpus and checkpoints present, the recorded phases were:

```bash
source .venv/bin/activate
python scripts/finalize_stage.py select
python scripts/finalize_stage.py evaluate
PYTHONPATH=src python -m unittest discover -s tests -v
```

Both result paths now exist and are protected against overwriting. To verify validation-only selection again, use `--selection runs/stage-close-check/selection.json`; this does not read test text. The test evaluation is already archived: inspecting it requires no new model comparison or test access. The evaluator accepts alternate output paths for deliberate reproducibility checks of the same frozen policy; that does not restore the book's untouched status.

For direct evaluation of the selected policy, the essential setting is `--context-length 16` with the selected checkpoint. Using default capacity 64 would evaluate a different policy. Generation likewise must cap its context at sixteen to match this finalized policy; the existing generation CLI defaults to checkpoint capacity, so it does not automatically apply the selection manifest. Temperature is a sampling choice, not part of these probability-based metrics.

All **43 tests pass**, including validation-only deterministic selection, exclusive result creation, rejecting inconsistent selection before model/test access, and the earlier modeling, training, evaluation and diagnostic checks. Source/checkpoint/archive hashes, selection timestamps, split sizes and target counts are recorded and verified. The model weights are unchanged.
