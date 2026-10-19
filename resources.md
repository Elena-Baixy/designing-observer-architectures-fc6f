# Resources Catalog

## Summary
Resources gathered for testing whether a trajectory-memory, multi-scale-future-latent, hazard-output **observer** beats a **direct activation verbalizer** (LatentQA / Activation Oracle / probe) trained on the same labels at forecasting unsafe and long-form behavior 100–2048 tokens ahead.

- **Papers:** 31 PDFs in `papers/`. Index: `papers/README.md`. Notes: `papers/notes/` (24 files).
- **Datasets:** 16 local datasets in `datasets/` (≈250 MB). Index: `datasets/README.md`.
- **Code:** 6 repos in `code/`. Index: `code/README.md`. The user-specified LatentQA repo was cloned and **smoke-tested**.
- **Synthesis:** `literature_review.md`, including the direction ranking (top 3 kept) in §8.
- **Helper scripts** in `scripts/`:
  - `arxiv_lookup.py`, `arxiv_search.py`, `arxiv_abs.py`: literature search.
  - `save_datasets.py`: dataset download.
  - `eda_horizons.py`: onset and length statistics.
  - `test_latentqa_reading.py`: LatentQA smoke test with compatibility shims.
  - `hf_check.py`: HF availability check.
  - `make_papers_readme.py`: builds `papers/README.md`.

## Papers
Total papers downloaded: 31

| Title | Authors | Year | File | Key info |
|---|---|---|---|---|
| LatentQA: Teaching LLMs to Decode Activations Into Natural Language | Pan, Chen, Steinhardt | 2024/26 | papers/2412.08686_latentqa.pdf | **User-specified**; LIT decoder, reads layer 15 and writes layer 0; future-completion QA; deep-read |
| Activation Oracles | Karvonen et al. | 2025 | papers/2512.15674_activation_oracles.pdf | General verbalizer; Qwen3/Gemma/Llama checkpoints |
| Building Better Activation Oracles | Bauer et al. | 2026 | papers/2606.02609_building_better_activation_oracles.pdf | On-policy, future-CoT QA, 5 layers, AObench |
| Confidence & Calibration of AOs | Torrielli et al. | 2026 | papers/2605.26045_activation_oracle_calibration.pdf | Candidate scoring gives AUROC 0.92–0.96 |
| Faithful Activation Verbalization (AVPO) | Zhao et al. | 2026 | papers/2609.34033_faithful_activation_verbalization.pdf | Hallucinations in verbalizers |
| Predictive Concept Decoders | Huang et al. | 2025 | papers/2512.15712_predictive_concept_decoders.pdf | Future-token pretraining improves harm forecasting |
| Patchscopes | Ghandeharioun et al. | 2024 | papers/2401.06102_patchscopes.pdf | Untrained patching baseline |
| SelfIE | Chen et al. | 2024 | papers/2403.10949_selfie.pdf | Untrained self-interpretation |
| Meta-Models | Costarelli et al. | 2024 | papers/2410.02472_meta_models.pdf | Behavior decoding |
| Future Lens | Pal et al. | 2023 | papers/2311.04897_future_lens.pdf | Future-token decay from one hidden state |
| Do LMs plan ahead? | Wu et al. | 2024 | papers/2404.00859_plan_ahead_future_tokens.pdf | Pre-caching vs breadcrumbs |
| Emergent Response Planning | Dong et al. | 2025 | papers/2502.06258_emergent_response_planning.pdf | Prompt states predict global attributes; U-shaped accuracy over the response |
| LLMs Decide Early and Explain Later | Datta et al. | 2026 | papers/2604.22266_decide_early_explain_later.pdf | Answer locked about 760–1900 tokens before the end |
| Predicting Output Length via Entropy-Guided Representations | Xie et al. | 2026 | papers/2602.11812_output_length_prediction_hidden_states.pdf | EGTP/PLP; ForeLen |
| StreamGuard (Predict, Don't React) | Kavumba et al. | 2026 | papers/2604.03962_streamguard_safety_forecasting.pdf | Monte Carlo rollout forecasting guardrail; OnTime metric |
| Streaming Content Monitor / FineHarm | Li et al. | 2025 | papers/2506.09996_streaming_content_monitor_fineharm.pdf | Token-level harm labels, Delay-k |
| Shallow Safety Alignment | Qi et al. | 2024 | papers/2406.05946_shallow_safety_alignment.pdf | Harm decided in the first ~5 tokens; prefill attacks |
| Refusal Is Mediated by a Single Direction | Arditi et al. | 2024 | papers/2406.11717_refusal_single_direction.pdf | Linear refusal features |
| Constitutional Classifiers | Sharma et al. | 2025 | papers/2501.18837_constitutional_classifiers.pdf | Streaming output classifier |
| Time-to-Unsafe-Sampling | Davidov et al. | 2025 | papers/2506.13593_time_to_unsafe_sampling.pdf | Survival framing plus conformal bounds |
| Forecasting Rare LM Behaviors | Jones et al. | 2025 | papers/2502.16797_forecasting_rare_behaviors.pdf | Deployment-scale risk forecasting |
| Hallucination Risk from Internal States | Ji et al. | 2024 | papers/2407.03282_hallucination_risk_internal_states.pdf | Pre-generation probe |
| Detecting Strategic Deception with Linear Probes | Goldowsky-Dill et al. | 2025 | papers/2502.03407_deception_linear_probes.pdf | Probe-monitor practice |
| Contrastive Predictive Coding | van den Oord et al. | 2018 | papers/1807.03748_contrastive_predictive_coding.pdf | InfoNCE multi-horizon |
| Multi-Token Prediction | Gloeckle et al. | 2024 | papers/2404.19737_multi_token_prediction.pdf | Token-space horizon limits |
| Belief State Transformer | Hu et al. | 2024 | papers/2410.23506_belief_state_transformer.pdf | Suffix encoder, belief states |
| Next-Latent Prediction (NextLat) | Teoh et al. | 2025 | papers/2511.05963_next_latent_prediction.pdf | Future-latent auxiliary objective |
| Hierarchical Latent Prediction (HiLP) | Shi et al. | 2026 | papers/2608.05806_hierarchical_latent_prediction.pdf | Multi-scale latent; gain grows with horizon |
| Nnet-survival | Gensheimer & Narasimhan | 2018 | papers/1805.00917_nnet_survival_discrete_time.pdf | Discrete-time hazard loss |
| Time-to-Event Prediction with NN & Cox (Cox-Time) | Kvamme et al. | 2019 | papers/1907.00825_time_to_event_nn_cox.pdf | pycox; C^td, IBS |
| Qwen3Guard Technical Report | Qwen Team | 2025 | papers/2510.14276_qwen3guard_technical_report.pdf | Token-level stream guard; Qwen3GuardTest |

## Datasets
Total datasets downloaded: 16 (14 via HF `save_to_disk`, 2 via GitHub CSV).

| Name | Source | Size | Task | Location | Notes |
|---|---|---|---|---|---|
| Qwen3GuardTest | HF `Qwen/Qwen3GuardTest` | 2,441 | Unsafe-onset spans (response, thinking) | datasets/qwen3guardtest/ | Only human-annotated onset indices; onsets early (median 7 / 47 tokens) |
| BeaverTails | HF `PKU-Alignment/BeaverTails` | 364k | Response safety | datasets/beavertails/ | Prompts for rollouts |
| Aegis 2.0 | HF `nvidia/Aegis-AI-Content-Safety-Dataset-2.0` | 33k | Prompt/response safety | datasets/aegis2/ | |
| ToxicChat | HF `lmsys/toxic-chat` | 10k | Real-user toxicity | datasets/toxic_chat/ | Out-of-distribution evaluation |
| Circuit-breakers | HF `abhayesian/circuit-breakers-dataset` | 4,476 | Harmful/refusal completions | datasets/circuit_breakers/ | Harmful completions ≈370 tokens |
| LLM-LAT harmful | HF `LLM-LAT/harmful-dataset` | 4,948 | Harmful/refusal completions | datasets/llm_lat_harmful/ | |
| HarmBench | GitHub CSV | 1,528 | Harmful behaviors | datasets/harmbench/ | Split by BehaviorID |
| StrongREJECT | GitHub CSV | 313 | Forbidden prompts | datasets/strongreject_csv/ | |
| AdvBench | HF `mlabonne/harmful_behaviors` | 520 | Harmful behaviors | datasets/harmful_behaviors_advbench/ | |
| JBB-Behaviors | HF `JailbreakBench/JBB-Behaviors` | 200 | Harmful/benign pairs | datasets/jbb_behaviors/ | |
| XSTest | HF `Paul/XSTest` | 450 | Over-refusal | datasets/xstest/ | False-positive control |
| harmless_alpaca | HF `mlabonne/harmless_alpaca` | 31k | Benign prompts | datasets/harmless_alpaca/ | Censored trajectories |
| Dolly-15k | HF `databricks/databricks-dolly-15k` | 15k | Benign instructions | datasets/dolly_15k/ | LatentQA stimuli |
| LongWriter-6k | HF `zai-org/LongWriter-6k` | 6,000 | Long-form writing (median ≈4.7k tokens) | datasets/longwriter_6k/ | Content forecasting; benign preambles |
| ForeLen (Qwen2.5-7B LongSeq) | HF `abinzzz/ForeLen` | 1,057 | Output length | datasets/forelen_qwen2.5_7b_longseq/ | Time-to-EOS |
| RealToxicityPrompts | HF `allenai/real-toxicity-prompts` | 99k | Continuation toxicity | datasets/real_toxicity_prompts/ | Short-horizon sanity check only |

See `datasets/README.md` for download commands, fields, samples, and horizon statistics.

## Code repositories
Total repositories cloned: 6

| Name | URL | Purpose | Location | Notes |
|---|---|---|---|---|
| LatentQA | github.com/aypan17/latentqa | **User-specified** verbalizer baseline (LIT) | code/latentqa/ | Smoke-tested; 4 compatibility shims in `scripts/test_latentqa_reading.py` |
| Activation Oracles | github.com/adamkarvonen/activation_oracles | AO training, evaluation, probe baseline | code/activation_oracles/ | Qwen3-8B AO checkpoint cached |
| Building Better AOs | github.com/japhba/activation_oracles | Improved AO recipe, AObench | code/building_better_activation_oracles/ | Needs vLLM for data generation |
| NextLat | github.com/JaydenTeoh/NextLat | NextLat / MTP / BST loss implementations | code/nextlat/ | Future-latent heads |
| Qwen3Guard | github.com/QwenLM/Qwen3Guard | Stream guard usage and evaluation | code/qwen3guard/ | `stream_moderate_from_ids` |
| pycox | github.com/havakv/pycox | Hazard losses, C^td / IBS metrics | code/pycox/ | Logistic-hazard = Nnet-survival |

## Resource gathering notes

### Search strategy
- The paper-finder service was **unavailable** (localhost:8000 not running), so the search was manual: arXiv API title lookups and keyword searches (`scripts/arxiv_*.py`) covering activation verbalizers, response planning / lookahead probes, streaming safety monitors, survival/hazard models for LLMs, and next-latent / multi-token prediction.
- Searches emphasized 2025–2026, since today is 2026-10-18 and many relevant papers postdate the model's training knowledge, e.g. StreamGuard, Building Better AOs, HiLP, Decide-Early.
- Deep reading was split across three parallel reading agents plus the main agent. LatentQA was read in full by the main agent.

### Selection criteria
1. Methods that serve as baselines (verbalizers and probes).
2. Evidence on how predictable the future is from activations as a function of horizon.
3. Streaming-safety task definitions, labels and metrics.
4. Architectural components (future latents, hazard losses).
- Datasets were chosen for (a) event-time labels and (b) prompt pools for constructing on-policy long trajectories.

### Challenges encountered
- Several key datasets are **gated** for the available HF token: FineHarm, shallow-vs-deep safety, WildGuardMix, WildJailbreak, HEx-PHI, walledai HarmBench/StrongREJECT/AdvBench. Substitutes: GitHub CSVs and mirrors.
- The `abinzzz/ForeLen` reasoning config fails to load (CSV schema mismatch); only LongSeq was saved.
- LatentQA's repo is stale relative to transformers 5 and imports a renamed function. Handled with non-invasive shims; the repo is untouched.
- **Measured horizon reality:** natural unsafe onsets are almost always within about 100 tokens. No public dataset has unsafe onsets at 512–2048 tokens.

### Gaps and workarounds
- **Long-lead unsafe events must be constructed:**
  - (a) Benign long-form preamble plus a late harmful sub-request (LongWriter prompt + HarmBench/StrongREJECT behavior), generated on-policy by Qwen3-8B or its abliterated variant.
  - (b) Long thinking traces where the event is the first unsafe token of the final answer, or answer lock-in.
  - (c) Benign long-form controls.
  - Token-level onsets are labeled with Qwen3Guard-Stream (same tokenizer). The labeler's onset accuracy can be validated on Qwen3GuardTest `response_loc` / `thinking_loc`.
- **Long-form content labels** come for free from the future text: MiniLM embeddings of future windows, or section/topic tags from an LLM judge.
- **Compute:** at check time, 3 of the 4 A6000s were ~43 GB occupied by other jobs; GPU 2 was free. Disk: ≈60 GB free on `/data`. Plan for one GPU and subsampled or compressed activations.

## Recommendations for experiment design

1. **Primary data:**
   - D1: constructed late-harm and thinking-trace trajectories from Qwen3-8B / Qwen3-8B-abliterated, with Qwen3Guard-Stream onsets.
   - D2: LongWriter-6k prompts with on-policy long outputs, for content forecasting.
   - Short-lead check and labeler validation: Qwen3GuardTest.
   - Controls: XSTest and harmless_alpaca.
2. **Baselines:** all trained on the same labels and activations.
   - LatentQA-style LoRA verbalizer (code/latentqa).
   - AO checkpoint fine-tuned and scored by closed-set logits.
   - Linear/MLP probe per horizon on the current position or the last-k pool.
   - Per-prefix value head (StreamGuard/SCM-like).
   - Text-only classifier.
3. **Observer:**
   - Causal transformer/GRU memory over compressed activations (stride 8–16, 1–3 layers, ≤512 dimensions).
   - Multi-scale future-window latent heads (scales 64/256/1024/2048, InfoNCE or SmoothL1 to stop-gradient targets).
   - Discrete-time logistic-hazard head with censoring.
4. **Metrics:**
   - AUROC/AUPRC per lead bin (key plot: gain vs lead distance).
   - Time-dependent concordance C^td; integrated Brier score / integrated negative binomial log-likelihood (pycox).
   - OnTime/Late/Miss and false-positive rate on benign responses.
   - R@k / cosine for content forecasting; Spearman for time-to-EOS.
5. **Code to reuse:**
   - `code/latentqa/lit/utils/activation_utils.py` (activation capture and injection).
   - `code/pycox` (losses and metrics).
   - `code/nextlat/models/*` (latent losses).
   - `code/qwen3guard` (labeler usage).
   - `scripts/test_latentqa_reading.py` (how to run LatentQA under transformers 5).
