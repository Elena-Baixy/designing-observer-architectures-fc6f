# Research State

- Current phase: `None`
- Pipeline completed: `True`

## Previous phases

resource_finder (succeeded), experiment_runner (succeeded)

## Current phase context

- Phase: `experiment_runner`
- Status: `completed`
- Started: `2026-10-19T01:44:34.770522Z`
- Next steps:
  - Validate the report and experimental artifacts before finalizing.

## Workspace check

- Root: `/data/smallyan/neurico/random_ideas/designing-observer-architectures-fc6f`
- Directory usable: `True`

## Output validation

- Valid: `True`
- Expected: `REPORT.md`
- Missing: None
- Outside workspace: None

## Agent notes

<!-- NEURICO_AGENT_NOTES_START -->
### resource_finder
<!-- NEURICO_AGENT_NOTES_START:resource_finder -->
**Phase:** resource_finder. All deliverables are on disk: papers/ (31 PDFs, README, notes/ ×24), datasets/ (16 datasets, ≈250 MB, README, .gitignore, horizon_stats.txt), code/ (6 repos, README), literature_review.md, resources.md, .resource_finder_complete.

**Key findings**
- The paper-finder service was down; the search was manual via arXiv API (scripts/arxiv_*.py).
- No prior work evaluates activation-based forecasting by lead distance. The longest horizons in the literature are ≤32 tokens for activation probes and ≤20 for next-latent methods.
- LatentQA (user-specified, deep-read) reads layer 15 and writes layer 0. It predicts properties of *future completions*, but has no time axis and no trajectory memory.
- **Natural unsafe onsets are early.** Qwen3GuardTest response_loc median is 7 tokens; thinking_loc median 47, max 307; 0% at ≥512 (datasets/horizon_stats.txt). **Long-lead (512–2048) events must be constructed** (literature_review.md §9).
- LatentQA repo smoke-tested on 1× A6000 (logs/latentqa_smoke_test.log). It needs 4 shims for transformers 5 (scripts/test_latentqa_reading.py); the repo itself is unmodified.
- Cached locally in $HF_HOME=/data/smallyan/hf_home: Qwen3-8B, Qwen3-8B-abliterated(-v2), AO checkpoint for Qwen3-8B, LatentQA Llama-3-8B decoder + NousResearch Llama-3-8B-Instruct, Qwen3Guard-Stream-0.6B/4B (token-level onset labeler, same tokenizer), all-MiniLM-L6-v2.
- **Gated / unavailable:** FineHarm, shallow-vs-deep safety data, WildGuardMix, WildJailbreak, HEx-PHI, walledai/*. Substitutes: GitHub CSVs and mirrors.

**Direction budget (scored in literature_review.md §8)**
- KEEP:
  - D1 (18/20): long-horizon unsafe event-time forecasting on constructed trajectories. These are late-harm multi-part long-form prompts plus thinking-trace → final-answer onsets, labeled by Qwen3Guard-Stream. Compare the observer with LatentQA/AO verbalizer and probes on the same labels, stratified by lead.
  - D2 (17): long-form content forecasting (future-window embeddings/topics, LongWriter-6k, on-policy).
  - D3 (17): component ablations (memory / multi-scale latent / hazard head) plus free-label time-to-EOS and answer-lock-in tasks.
- REJECTED:
  - D4: training new AOs from scratch (compute).
  - D5: WildChat/LMSYS multi-turn (off-policy, gated, noisy).
  - D6: RealToxicityPrompts (horizon ≈20 tokens).
  - D7: steering (out of scope).
  - D8: conformal bounds (kept only as an optional add-on).
  - D9: scale study (compute).

**Next phase (experiment_runner): concrete steps**
1. Use .venv (torch 2.14 cu130, transformers 5.20, peft, datasets, sklearn). Target model Qwen3-8B on the free GPU (check nvidia-smi; only GPU 2 was free).
2. Build the D1 dataset. Prompts: LongWriter preamble + HarmBench/StrongREJECT/AdvBench harmful sub-request (split by behavior ID), plus benign controls. Generate up to about 2.5k tokens on-policy with Qwen3-8B-abliterated or Qwen3-8B (and thinking mode). Label per-token onset with Qwen3Guard-Stream-4B (`stream_moderate_from_ids`), validating the labeler on Qwen3GuardTest.
3. Cache activations: layers about 18/27 of 36, stride 8–16, optionally PCA to 512 dimensions; mind the ≈60 GB disk limit.
4. Train the observer (memory + multi-scale future-latent heads + logistic-hazard head with censoring) and the baselines on the same labels:
   - LatentQA-style LoRA verbalizer, scored by closed-set logits;
   - AO checkpoint fine-tuned;
   - linear/MLP probe per horizon;
   - per-prefix value head;
   - text-only classifier.
5. Report AUROC per lead bin [1,64)…[1024,2048), C^td and integrated Brier score (pycox), and OnTime/Late/Miss. The key plot is the gain vs lead distance.

**Uncertainties**
- Whether constructed late-harm onsets are realistic, and whether the abliterated model complies with late sub-requests reliably.
- Qwen3Guard-Stream labeler noise at long contexts.
- Censoring is informative.
- GPU contention.
<!-- NEURICO_AGENT_NOTES_END:resource_finder -->

### experiment_runner
<!-- NEURICO_AGENT_NOTES_START:experiment_runner -->
**Phase:** experiment_runner. Complete: planning.md, src/*, results/, figures/, REPORT.md and README.md are on disk.

**What was done**
- 96 unmodified benchmark prompts (LongWriter, ToxicChat, BeaverTails, Aegis 2.0, XSTest), split by family before generation.
- Qwen3-8B generation up to 2048 tokens.
- gpt-5.6-luna timeline annotation, called through OpenAI directly because no OpenRouter key was available.
- Layer 9/18/27/35 block-activation cache.
- Compared: direct_AO, A horizon-query, B latent rollout, D hierarchical, plus 4 ablations and zero/shuffle controls, 3 seeds each. Events: unsafe and caution.

**Key findings**
- Unsafe-onset forecasting is unpowered: 5 natural onsets, all ≤448 tokens. No significant differences.
- Caution event: A and D beat direct_AO by +0.03–0.11 AUROC (n.s.). The gain does not grow with H. Pooled AUROC is largely prompt-source separation (results/per_family.md).
- Memory helps at long H. JEPA, the hazard head and rollout showed no benefit.
- Content forecasting works: behavior 0.85–0.88 at ≤1024 tokens ahead; summary MRR about 0.10 vs 0.019.

**Deviations**
- The plan to construct late-harm prompts was dropped: we do not elicit harmful content.
- The first 216-prompt run ran out of GPU memory and was discarded.
- No off-the-shelf AO/LatentQA baseline, no repeated rollouts, no LLM-judged summaries (1-hour budget).

**Next**
- Scale data 10–20×.
- Find natural late-onset regimes (thinking traces, multi-turn).
- Add an AO decoder baseline.
- Fix hazard-head calibration at long H.
<!-- NEURICO_AGENT_NOTES_END:experiment_runner -->

<!-- NEURICO_AGENT_NOTES_END -->
