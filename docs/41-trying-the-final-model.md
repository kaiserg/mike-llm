# Trying the finalized small model

[Documentation index](README.md) · [Previous](40-modern-longform-expansion-wave5.md)

The finalized checkpoint can generate continuations locally on CPU. It is a 17,232-parameter character model trained on *Frankenstein*, with a 16-character active inference window selected on validation. It predicts one lowercase character, space or newline at a time. It is not an instruction-following or question-answering model.

The convenience driver [`scripts/try_final_model.py`](../scripts/try_final_model.py) reads the frozen selection manifest, verifies the checkpoint SHA-256, loads its vocabulary and automatically applies the selected context window.

## Quick start

From the repository root:

```bash
source .venv/bin/activate
python scripts/try_final_model.py
```

The default experiment uses prompt `the creature `, temperature 0.7, seed 42 and 200 new characters. To supply another prompt:

```bash
python scripts/try_final_model.py \
  --prompt "i felt " \
  --temperature 0.7 \
  --max-new-tokens 200 \
  --seed 123
```

Prompts may contain lowercase English letters, spaces and newlines. Uppercase letters, punctuation, digits and accented characters are absent from this checkpoint's vocabulary and are rejected rather than silently changed.

## What the controls mean

| Option | Effect |
| --- | --- |
| `--prompt` | Text that the model continues. Only its most recent 16 characters remain visible during later generation. |
| `--max-new-tokens` | Exact number of new characters to produce. There is no end-of-sequence token. |
| `--temperature 0` | Greedy, deterministic generation. It often repeats a locally likely phrase. |
| `--temperature 0.5` to `0.8` | Samples among plausible next characters and usually gives a more varied demonstration. |
| `--seed` | Makes positive-temperature sampling reproducible. |

Temperature changes how the logits are sampled; it does not retrain the model or add knowledge.

## Verified examples

With the default prompt and seed, temperature 0.7 began:

```text
the creature when i not and on live the beings the affecture undered from sticling...
```

The output contains English-like local fragments but does not maintain a reliable meaning. Greedy generation with temperature 0 quickly entered this loop:

```text
the creature and the should and the should and the should...
```

These are expected limitations of the experiment. The model learned local character and word patterns from one novel; it did not learn a dialogue format, broad facts, instruction following or dependable answers.

## Use the general generator

The lower-level command supports any compatible checkpoint. For the finalized model, pass the selected 16-character window explicitly:

```bash
python src/generate.py \
  --checkpoint runs/position-exposure-40000/seed-42/primary/checkpoint.pt \
  --context-length 16 \
  --prompt "the creature " \
  --max-new-tokens 200 \
  --temperature 0.7 \
  --seed 42
```

The convenience driver is safer for this checkpoint because it verifies the frozen manifest and fills in the selected inference context automatically.

[Documentation index](README.md) · [Previous](40-modern-longform-expansion-wave5.md)
