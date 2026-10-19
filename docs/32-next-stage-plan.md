# Next stage: a broader licensed corpus and a stronger base model

[Documentation index](README.md) · [Previous](31-stage-closure-and-test.md) · [Next](33-corpus-ingestion-pilot.md)

**Status: proposed implementation plan, not a completed corpus or training experiment.** The prior stage is closed; its selected model and consumed test remain historical references. This plan does not download documents, assign actual new test items, train models or evaluate any new test. Targets below are starting design choices, not measured results or runtime promises.

## Objective and scope

Build a more general **English base language model** on diverse, attributable text before instruction or question-answer training. English is the working default to preserve continuity; a Spanish or bilingual objective would require an explicit corpus mixture, separate language metrics and corresponding evaluation groups. The user-facing language of this project does not determine the model's training language.

Success means better next-token prediction on multiple validation domains, more coherent continuations and a reproducible data/model pipeline. It does not mean that a low perplexity automatically makes a reliable assistant. Do not add instruction tuning yet, and do not use the old test to choose the new model.

## 1. Register sources, rights and document families

Start with a small reviewed ingestion pilot, then grow a versioned corpus:

| Milestone | Proposed scope | Purpose |
| --- | --- | --- |
| Ingestion pilot | 10–20 documents; approximately 1–3 million normalized characters if available | Verify extraction, attribution, deduplication and deterministic manifests |
| Base-model pilot | Approximately 10–30 million normalized characters; at least 100 documents and roughly 60 independent author/source-lineage groups | Support several domains and multiple held-out groups without immediately requiring a large model |
| Expansion | Set after measuring pilot throughput and quality | Scale only when data quality and validation justify it |

These are planning ranges. Report actual UTF-8 bytes, characters, tokens, documents, author groups and split proportions. Long books must not silently dominate a nominally diverse corpus.

Proposed training mixture by normalized text volume: **40% fiction, 30% narrative nonfiction/history/essays, 30% explanatory/educational prose**. Aim for at least three domains, multiple periods and many authors, with at least two independent source families when suitable rights-cleared texts are available. Cap one author at a proposed 10% of training bytes; log exceptions rather than concealing them through document counts. Mixture weights and caps are frozen using training-side metadata before the comparison runs.

### Source shortlist, reviewed 2026-10-19

| Source | Contribution | Rights evidence and intake decision |
| --- | --- | --- |
| Project Gutenberg | Fiction, essays, history and other nonfiction across authors | Candidate source; item-level editions and territorial status require checking. Its public-domain statements concern the US; the license distinguishes underlying text from Gutenberg material. Preserve original notices and record the specific edition. [License](https://www.gutenberg.org/policy/license), [permission guidance](https://www.gutenberg.org/policy/permission) |
| Standard Ebooks | Carefully produced literary editions and nonfiction where available | Candidate alternative edition source. The project describes its books as free of US copyright restrictions and its own work as public-domain. Verify each work and applicable territory. It often overlaps Gutenberg: another edition is not an independent text. [Official description](https://standardebooks.org/) |
| Open Logic Project | Modern expository logic/formal-methods prose | Priority small educational pilot: the project states CC BY 4.0 except where otherwise noted. Review the exact files and exceptions, preserve attribution, and distinguish prose from formulas/source markup. [License](https://openlogicproject.org/olp-license/), [official repository](https://github.com/OpenLogicProject/OpenLogic/) |
| Wikimedia text | Broader encyclopedic topics | Optional separately tracked component, not an automatic bulk ingest. Wikimedia's terms generally specify CC BY-SA 4.0 for text with attribution and other provisions; exact revisions, imported-text exceptions and historical dump notices matter. Record revision/license and retention of obligations. [Terms](https://foundation.wikimedia.org/wiki/Terms_of_Use) |
| OpenStax | Broad explanatory textbooks | Defer from the default permissive pilot. Current official guidance states CC BY-NC-SA, so do not assume the older CC BY label applies to a new download. A particular snapshot requires its own license record and a deliberate compatible use/distribution policy. [Official licensing guidance](https://help.openstax.org/s/article/Licensing-information-of-OpenStax-textbooks), [FAQ](https://openstax.org/faq) |

Use explicit public-domain status/CC0 or CC BY sources first where suitable; keep share-alike material distinguishable until its obligations and intended release are resolved. Do not infer text rights from a hosting platform's software license, an “open” label or download access. This shortlist is not blanket clearance of all files, worldwide copyright status, model-weight redistribution or third-party assets. Restrict the initial pipeline to reviewed text; do not import images or arbitrary web pages by association.

Each accepted document needs a record with: stable document/work ID; title; author/contributors; language/domain; edition/revision; source and download URLs; retrieval time; raw SHA-256; license identifier and exact rights-evidence URL/text; territory if relevant; attribution; exceptions; transformation version; normalized SHA-256; author/work/translation/edition family IDs; acceptance status and reason. Keep rejected records and reasons too. A record with unresolved rights stays out of the prepared training corpus.

Deliverables: `configs/corpus_v2_sources.json`, local raw manifest, retained notices, `ATTRIBUTION` data and an acceptance report. These are **planned artifacts**, not configuration files already understood by the current downloader.

## 2. Freeze a genuinely new multi-document test

Treat the previous three books—Frankenstein, Alice in Wonderland and Pride and Prejudice—and their closely related editions/adaptations/translations as a **legacy exclusion family** for the new benchmark corpus. Keep old checkpoint results available as history; exclude these families from all new stage-2 splits to avoid calling familiar material a fresh test.

Split complete related-document groups **before** tokenization, packing or training windows:

1. Resolve work/edition/translation identities and cluster exact and near duplicates across candidate sources. For author-generalization, keep the author's works in one split; connect related translations/coauthored families conservatively.
2. Assign fixed group IDs to train/development/new test, stratified by domain and, where feasible, source and period. Target approximately 80/10/10 by groups, with a proposed minimum of eight independent groups in each held-out split and representation from all three domains. If minimum coverage changes the ratios, publish the actual group and byte ratios. Do not rebalance by copying chapters across splits.
3. Freeze an explicit group-to-split manifest and its hash. Adding later documents must not reshuffle existing assignments: assign new families against the frozen manifest or create a new dataset version. The existing hash-sort splitter does not provide incremental stability automatically.
4. Remove or assign cross-split duplicate components as a unit **before freezing**. Record exact/near-duplicate findings and deterministic resolution rules. A starting near-duplicate screen may use normalized word 5-shingles, MinHash candidate retrieval and exact Jaccard confirmation at a proposed 0.8 threshold, plus repeated-passage checks; tune the rule on training/pilot material and version it. Such a threshold is a screening heuristic, not proof of no contamination.
5. Restrict human text inspection and exploratory quality tuning to training/development. Automated test ingestion, hashing, decoding, duplication checks and fixed quality validation are necessary to construct the split, but must not become model/loss-based test selection. Logs should expose metadata/status, not held-out passages or intermediate model scores.
6. Reserve the new test for **one frozen model/policy selection at the end**. No test-based tokenizer choice, mixture changes, prompt selection, head-mask search or checkpoint comparison. Write the selection manifest first, then report the chosen model on the complete test and its predetermined domain groups.

If contamination or pipeline defects are discovered after freezing, invalidate that dataset version transparently and create a replacement before training/model selection; do not silently patch test examples based on model performance.

Deliverables: versioned split manifest, group-overlap/duplicate report, train/development reports, test metadata and checksums, and a test-access ledger. **No new test is actually reserved yet**; that milestone is reached only when real accepted document IDs and assignments are frozen.

## 3. Preserve text and establish a robust tokenizer baseline

The old lowercase 28-character corpus is too restrictive for the intended broader base model. Preserve case, punctuation, paragraph structure and useful Unicode. Specify UTF-8 decoding, Unicode normalization, line-ending/whitespace rules and boilerplate removal; retain raw originals separately. Never apply a generic markup stripper that silently deletes logical formulas or explanatory content. Decide how formulas are rendered and test representative training examples.

Use **UTF-8 bytes as the first new tokenizer baseline**, with an explicit end-of-document token. This avoids fitting a vocabulary on held-out text or failing on previously unseen Unicode characters. It is a proposed new implementation: the current tokenizer and checkpoints remain character-level. Byte contexts count bytes, not characters; report that unit everywhere. Test round trips, multibyte characters, document boundaries, loss masks and decoding of incomplete generated byte sequences.

Train new byte models from random weights: the old embedding/output vocabulary does not transfer directly. Keep a simple train-counted unigram baseline and a small byte Transformer as reference. Do not use old character PPL as a numerical score target for a differently normalized/tokenized corpus.

Only after that baseline is stable, consider a separate train-only BPE experiment (for example vocabularies around 2000–4000). Freeze tokenizer files/hashes and special IDs; document how differing context spans, sequence lengths and training token counts affect comparisons. Prefer **bits per UTF-8 byte** on the same evaluation text as the cross-tokenizer metric, with full-stream target coverage and conditioning protocol explicit; raw token perplexities are not comparable across tokenizers.

Deliverables: Unicode-preserving preparer, byte tokenizer, document-boundary-aware dataset/windows and tests. Extend the current pipeline without changing historical corpus files or pretending old checkpoints support the new vocabulary.

## 4. Improve the base model in controlled, affordable steps

First profile training throughput, peak memory and checkpoint/evaluation time on the actual machine. Keep the reproducible CPU reference. Test MPS separately if useful, recording device/library differences rather than assuming CPU bitwise reproducibility carries over. Set concrete run budgets from those measurements; the corpus-size proposal is not permission for unbounded runs or an unmeasured runtime estimate.

A starting ladder to validate through the pilot:

| Step | Proposed experiment | Control |
| --- | --- | --- |
| Reference | Existing simple architecture adapted to bytes and document boundaries | Freeze corpus/tokenizer and a small reproducible baseline |
| Capacity | Width 64, 2 blocks, 4 heads; then width 128, 4 blocks, 4 heads if throughput permits | Keep tokenizer/context fixed; report parameter counts and processed bytes |
| Context | For the selected affordable width/depth, compare 64/128/256 **byte-token** windows | Same optimizer recipe and processed training-byte budget; report updates and per-position exposure |
| Position handling | Keep learned positions as the first default; test an alternative only with a specific failure hypothesis | No simultaneous tokenizer/capacity/context change |
| Optimization | Short learning-rate pilot, then a fixed schedule with warmup/decay if supported by development results | No test feedback; retain stable gradient clipping and checkpoint provenance |

Use fixed seeds **42, 7 and 123** for the short-listed comparisons, recording means and sample variation. Avoid testing every combination at once. Preserve optimizer/RNG state for exact same-environment resume; compare a resumed run with an uninterrupted small reference before relying on it. Check document boundaries and prevent padding/concatenation from introducing false targets or cross-split context.

Track updates, non-padding target tokens, UTF-8 training bytes, effective epochs, wall time, device and peak memory. Within the byte baseline, use equal processed training-byte budgets for architecture comparisons and separately report performance under a fixed wall-time budget if measured. Across tokenizers, matching token counts alone is not a matched data or compute budget. Log exposure counts at absolute positions; longer contexts previously changed exposure per position.

Do not port the selected diagnostic masks as the default model recipe. Those effects were observed in two small checkpoints under a particular sixteen-token isolation, not a general architectural recommendation.

## 5. Development metrics and an explicit gate before QA

Choose using development only:

- Pooled loss and **bits per byte** across the fixed development stream, per-domain scores, a macro average over document groups and individual-group spread. Report top-1 accuracy and processed target counts as complementary metrics.
- Training/development curves and the gap between them; reproducibility/resume checks; throughput and memory.
- Fixed training/development-side prompts for narrative and explanatory continuations. Use fixed generation seeds and temperatures (for example greedy, 0.8 and 1.0); keep every sample, not just favorable continuations. Review punctuation, coherence and repetition with a small predetermined rubric.
- Repeated passage/long-copy comparisons against training text as a memorization diagnostic, with the method recorded. These checks do not establish privacy guarantees or factual reliability.

Proposed promotion criterion: a short-listed model improves mean development bits/byte by at least **5% relative to the new small byte baseline**, improves in at least two of three seeds, and has no domain worse by more than a proposed **2%** at the comparable data budget. These are initial decision thresholds to freeze **before** the comparison, not promised outcomes. If the candidate fails, keep the baseline and investigate the training/development data or recipe; do not expand test access. Qualitative continuation and correctness tests must also pass; there is no universal perplexity threshold for QA readiness.

Select the configuration by the predetermined aggregate development objective, then select one checkpoint/seed by development loss with an explicit stable tie-break rule. Freeze inference window, tokenizer, checkpoint hash and generation defaults. Only then evaluate the single selected model on the newly reserved base-model test, reporting overall and per-domain metrics without revising the choice. Report the seed-selection and repeated-development bias plainly.

## 6. QA is a later, separate stage

After the base-model gate and closure:

- Define the desired language, answer style and task: start with **short, source-grounded reading questions**, not an unrestricted general assistant.
- Build licensed instruction/answer examples with provenance, using training-source documents. Synthetic answers require verification; being generated does not ensure truth or clearance.
- Reserve separate QA development/test families and evaluation questions before instruction training, avoiding answer/near-duplicate leakage. The base-model text test and a QA test measure different tasks; neither should be repeatedly used to tune the other.
- Introduce a conversation format and supervised answer loss deliberately. Evaluate answer correctness, source support, appropriate inability to answer, repetition and retention of base language ability. Keep the prior base model as the control.

No instruction examples, fine-tuning or assistant claims are introduced in this plan's implementation phase. A broader base corpus alone does not provide reliable factual question answering.

## Prioritized backlog and completion evidence

| Priority | Work item | Evidence required before proceeding |
| --- | --- | --- |
| P0 | Source registry and ingestion pilot | Accepted/rejected records, rights evidence, raw/normalized hashes and attribution; sample extraction review on training-side material |
| P0 | Grouping, deduplication and explicit splits | Frozen new document/group IDs, old-family exclusions, split hashes, domain coverage and no group overlap |
| P1 | Unicode/byte data pipeline | Round-trip and boundary tests, deterministic regeneration and target-count audits |
| P1 | Baseline and hardware profile | Repeated small run, resume equivalence, development metrics and measured resource budget |
| P2 | Limited capacity/context comparisons | Fixed data/tokenizer/budget, three seeds, all results and generation samples retained |
| P2 | Freeze winner and evaluate new test | Selection timestamp/hash precedes test scores; single selected policy and per-domain report |
| Later | Source-grounded QA | Base-model gate passed and a separate licensed QA/evaluation protocol frozen |

**First actionable milestone:** create the versioned source registry and ingest a small rights-reviewed, multi-domain pilot; implement group IDs and a stable split manifest before selecting or reading a model test score. Do not extend the old three-URL downloader into an uncontrolled crawler. No new data is acquired by merely saving this plan.

## Pilot implementation update

The [first ingestion pilot](33-corpus-ingestion-pilot.md) now registers and ingests fourteen development-only documents. Twelve books have extracted Unicode text; two LaTeX sections are quarantined. The new test is still not assembled, and inspected pilot families are ineligible for it. The remaining stages above are planned work.

The [editorial cleaning and near-duplicate screen](34-editorial-cleaning-and-near-duplicates.md) now provides a second immutable development snapshot. Explicit rules changed four books and removed 5,773 characters while preserving attribution; eight books remain byte-for-byte identical. No document pair or screened cross-document paragraph reached the frozen 0.80 Jaccard threshold. This completes the pilot cleaning screen, but broader source acquisition, source-family balance, split grouping and a future frozen test are still pending.

The [first modern educational expansion wave](35-modern-educational-expansion-wave1.md) adds fifteen CC BY 4.0 Carpentries chapters across five technical topics. Five pinned repositories yield 241,149 rendered characters but count as one conservative split group because they share a rights holder, template and contributor ecosystem. Three reused example/output paragraphs are documented. This improves topic and date coverage without satisfying the broader independent-source target; no split or new test has been created.
