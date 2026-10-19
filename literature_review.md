# Literature Review: Observer Architectures for Long-Horizon Forecasting of Unsafe and Long-Form LLM Behavior

Detailed per-paper notes: `papers/notes/<arxiv_id>.md` (24 notes; LatentQA, Activation Oracles, StreamGuard, SCM/FineHarm, Emergent Response Planning, NextLat/HiLP were deep-read in full).

## 1. Research area overview

The hypothesis sits where four literatures meet:

1. **Activation verbalizers.** Decoders that read an LLM's activations and answer questions in natural language: Patchscopes, SelfIE, LatentQA, Activation Oracles (AOs), Predictive Concept Decoders (PCDs), Meta-Models.
   - All read a *fixed set of positions* (typically ≤50 tokens) from one or a few layers.
   - All answer about the *present* or the *whole next reply*.
   - None is trained or evaluated with an explicit time axis.
2. **Lookahead and planning probes.** Hidden states encode the future: Future Lens, "Do LMs plan ahead", Emergent Response Planning, length prediction from hidden states (EGTP/PLP), "LLMs decide early and explain later", hallucination-risk probes.
   - Predictability decays quickly with distance (Future Lens: 1–3 tokens).
   - Probes on the prompt state still predict *global* response attributes (length, final answer), with Spearman ≈0.8.
   - Accuracy is U-shaped over the course of the response.
3. **Streaming safety monitors.** Monitors that act on partial generations: Constitutional Classifiers (streaming output classifier), Qwen3Guard-Stream, SCM/FineHarm, StreamGuard (forecasting by Monte Carlo rollouts).
   - All operate at harm onsets within the first sentence or about 100 tokens.
   - None reports performance as a function of *lead distance*.
4. **Future-latent prediction and survival analysis.** Methods supplying the observer's components: CPC/InfoNCE, Multi-Token Prediction, Belief-State Transformer, NextLat, Hierarchical Latent Prediction (HiLP) for multi-scale future latents; Nnet-survival / Cox-Time / DeepHit for discrete-time hazards; conformal time-to-unsafe-sampling bounds.

**Central gap:** across the 31 papers, no work forecasts *when* a behavior will occur, at leads of 100–2048 tokens, from activations. The longest activation-based lookahead found is about 32 decoding steps (PLP); next-latent methods reach about 20 tokens.

## 2. Key papers

### 2.1 Activation verbalizers (the baselines to beat)

**LatentQA / LIT** (Pan, Chen, Steinhardt; 2412.08686, ICLR 2026). *This is the user-specified paper and repo.*
- **Method:** decoder = LoRA copy of the target (r=32, α=64, lr 1e-4, batch 128).
  - It reads residual activations at layer k=15 of Llama-3-8B-Instruct, about 50 positions.
  - These replace the input embeddings of "???" placeholder tokens at decoder layer 0. A sweep found k=15, ℓ=0 best (eval ppl 1.013).
- **Data:** 16,732 o1-generated (control, stimulus, completion, QA) items in 3 types: goals, personas, extractive QA. Control activations are masked so the decoder must infer the control from stimulus activations. The data ships in the repo.
- **Framing:** the task is explicitly "predict qualitative properties of *future* completions from *current* prompt activations".
- **Results:**
  - Hidden-persona identification from user-turn activations: 100/92% (easy/hard), vs GPT-4 reading the full dialog at 97/73 and SelfIE at about 23/16.
  - Relational extraction: +32 points over linear probes.
  - CrowS-Pairs debiasing: the only method with a significant effect.
  - Steering works on unseen personas.
  - Gains scale with model and data size.
- **Limitations:** no temporal output, a static read window, whole-completion labels, hallucination risk.
- **We verified it runs:** `scripts/test_latentqa_reading.py`, log in `logs/latentqa_smoke_test.log`.

**Activation Oracles** (Karvonen et al.; 2512.15674).
- **Method:** LoRA (r=64) copy of the target. It reads activations at 25/50/75% depth, injected additively and norm-matched after layer 1 onto " ?" tokens: h' = h + ||h||·v/||v||.
- **Training:** about 1M examples (LatentQA system-prompt QA, 7 binary classification tasks, self-supervised past/next-token "context prediction" on FineWeb and LMSYS).
- **Models:** Qwen3-8B, Gemma-2-9B, Llama-3.3-70B, Claude Haiku 3.5. About 10 H100-hours for Qwen3-8B.
- **Results and caveats:** strong out-of-distribution auditing (Taboo secret word 0.205 vs 0.04 for white-box baselines), but uncalibrated.
- **Released checkpoints:** e.g. `adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B` (cached locally) and Qwen3-1.7B/4B/14B/32B variants.

**Building Better AOs** (2606.02609).
- **Changes:** on-policy rollouts, a CoT-chunk conversational QA set (questions about the *future* part of a reasoning trace, asked from earlier activations), 5 adjacent layers (21–25), stronger injection.
- **Result:** AObench score +0.244 → +0.435.
- **Advice:**
  - Score yes/no answers by AUROC on the logit margin, not accuracy.
  - Use ≥20–50 read positions.
  - Narrow fine-tuning of AOs only matched linear probes.
  - A text-only baseline was competitive in early versions.
- **Checkpoint:** `ceselder/qwen3-8b-ao-v3-best`.

**AO calibration** (2605.26045).
- When candidates can be enumerated, having the oracle score each candidate gives AUROC 0.92–0.96.
- Self-reported confidence is useless (AUROC ≈0.5).
- **Implication:** our verbalizer baseline should score a closed set of options, e.g. lead-distance bins, by likelihood. This turns it into a hazard-like distribution and makes the comparison fair.

**Predictive Concept Decoders** (2512.15712).
- **Method:** layer-15 activations → top-16 sparse concepts out of 32k → written to layer 0 of a LoRA decoder. Pretrained to predict the *next 16 tokens* from 16 activations, then fine-tuned on QA.
- **Result:** detects upcoming harmful output from jailbreak-prompt activations about 31% of the time, vs 14% for LatentQA. This is evidence that *future-prediction pretraining helps verbalizers forecast*. No public checkpoint.

**Patchscopes, SelfIE, Meta-Models** (2401.06102, 2403.10949, 2410.02472). Untrained or lightly trained patching decoders. These are weaker baselines; LatentQA beats Patchscope by about 38 points.

### 2.2 Lookahead / planning from hidden states

**Future Lens** (2311.04897, GPT-J-6B).
- Linear maps to future tokens decay fast: precision@1 of 29.2 → 15.8% at +1 to +3 tokens.
- A learned soft-prompt decoder holds about 44–48%.
- **Lesson:** compare against a strong learned decoder, not just a linear probe.

**Emergent Response Planning** (2502.06258).
- An MLP probe on the prompt's last hidden state predicts response length (Spearman 0.80–0.85 on instruct models), number of reasoning steps, story character choices, multiple-choice answers, and confidence.
- Probe accuracy is U-shaped across the response: lowest mid-response, where trajectory memory should help most.
- Responses are capped at 1000 tokens. The datasets are HF-hosted (UltraChat, GSM8K, TinyStories, CommonsenseQA, ...).

**EGTP / PLP output-length prediction** (2602.11812).
- Pooling hidden states weighted by next-token entropy beats last-token or mean pooling (MAE 103 vs 120–155).
- Progressive remaining-length prediction improves as decoding proceeds, but only up to 32 steps are reported.
- Releases **ForeLen** (`abinzzz/ForeLen`), with responses of 0.6k–2.2k tokens on average and up to about 8k.
- Time-to-EOS is a free event-time label.

**Decide Early, Explain Later** (2604.22266).
- In Qwen3-4B reasoning traces, the final answer is fixed while about 760–1900 tokens of reasoning remain.
- A memoryless per-sentence linear probe saves about 500 tokens at under 2% accuracy loss.
- Answers flip about 3 times per trace.
- This is a natural long-lead event: when the answer locks in.

**Do LMs plan ahead** (2404.00859) and **Belief State Transformer** (2410.23506).
- Future information is largely a by-product ("breadcrumbs") in small models; deliberate pre-caching grows with scale.
- Independent per-horizon heads need not form a belief state. A suffix encoder used only during training supplies gradient from the whole future, which motivates training-time future encoders.

**Hallucination risk from internal states** (2407.03282). A query-only SwiGLU probe reaches 84%, but transfer across tasks is weak and responses are short.

### 2.3 Streaming safety monitors (task framing, labels, metrics)

**StreamGuard** (2604.03962).
- Moderation as *forecasting*: a value head on every prefix predicts the expected harmfulness of future continuations.
- Supervised with Monte Carlo rollouts: 16 samples at temperature 0.7, judged by Qwen3Guard. Positions are labeled densely for the first 64 tokens, then every 4th.
- Results on QwenGuardTest `response_loc` (8B): OnTime 92.6% vs 89.9% for Qwen3Guard-Stream; output F1 81.9 vs 80.4.
- Rollout-model choice matters a lot.
- Onsets are in the first sentence 91% of the time. There is no lead-distance analysis.

**Streaming Content Monitor (SCM) / FineHarm** (2506.09996).
- 29K responses (about 14K harmful) with sentence-level and part-of-speech word-level harm labels, from WildGuardMix/WildJailbreak plus an uncensored Llama-3.1-8B.
- Model: a token-level head plus a holistic head with a logic loss; detection triggers after k flagged tokens (Delay-k).
- Average stop is at 18% of the response (about 100 tokens). Harmful responses average about 550–600 tokens.
- The HF dataset `liyang-ict/FineHarm` is **gated**; our token lacks access.

**Constitutional Classifiers** (2501.18837). The streaming output classifier blends per-token and cumulative-max predictions during training; metrics include leak rate and TPR at fixed FPR on benign traffic.

**Shallow safety alignment** (2406.05946).
- Alignment differences between aligned and base models are concentrated in the first ≈5 tokens.
- Prefilling k harmful tokens raises attack success to about 50% by k≈10.
- Their "safety recovery" data creates late refusals, which give non-monotone events.

**Time-to-unsafe-sampling** (2506.13593). A survival framing with conformal lower predictive bounds; "time" counts resampling attempts, not tokens. Their calibration can be layered on our hazard head.

**Forecasting rare behaviors** (2502.16797). Extrapolates elicitation probabilities across query volume. Not about token time, but its tail-forecasting mindset is relevant.

**Qwen3Guard tech report** (2510.14276).
- Qwen3Guard-Stream is a token-level safe/controversial/unsafe classifier sharing the Qwen3 tokenizer.
- **Our best automatic event-time labeler:** the 0.6B and 4B models are cached locally.
- The paper also releases `Qwen/Qwen3GuardTest`, which we downloaded.

### 2.4 Components for the observer

**CPC** (1807.03748). InfoNCE with a separate W_k per horizon:
L_k = −E log[exp(z_{t+k}ᵀW_k c_t) / Σ_j exp(z_jᵀW_k c_t)].
log N − L_k lower-bounds I(z_{t+k}; c_t), which can serve as a per-horizon diagnostic of how much future information remains.

**Multi-Token Prediction** (2404.19737). Gains peak at n≈4 tokens. The authors suggest embedding-space targets for longer horizons.

**NextLat** (2511.05963, code cloned).
- Auxiliary SmoothL1 prediction of the next latent from (latent, next token), with stop-gradient targets.
- Provably converges toward belief states.
- Beats MTP out to 20 tokens; the authors name *hierarchical* belief states as future work.

**HiLP** (2608.05806). Adds a higher-level, window-pooled abstract latent. Its advantage over NextLat grows with horizon (up to 8 steps). This is the closest precedent for our "multi-scale future window latents".

**Discrete-time hazard: Nnet-survival** (1805.00917).
- Per bin j with hazard h_j and survival S_j = ∏_{i≤j}(1−h_i).
- Event observed in bin j: log L = ln h_j + Σ_{i<j} ln(1−h_i).
- Censored in bin j: log L = Σ_{i<j} ln(1−h_i).

**Cox-Time and DeepHit** (1907.00825; pycox cloned). Cox-Time is a sampled Cox loss; DeepHit adds a ranking term (best ranking, worst calibration).
- Metrics: time-dependent concordance C^td, and the IPCW integrated Brier score / integrated binomial log-likelihood (pycox `EvalSurv`).

## 3. Common methodologies

- **Injection-style verbalizers.** Activations are placed at placeholder tokens: replaced at layer 0 (LatentQA, PCD) or added with norm-matching after layer 1 (AO). LoRA on a copy of the target; read layers in the middle or at 25/50/75% depth; ≤50 positions.
- **Probes.** Linear or 1-hidden-layer MLP on a single position (the last prompt token) or on pooled positions (mean, max, entropy-weighted).
- **Per-prefix heads for streaming.** A linear or value head on every token; labels from final-response labels (copied to every prefix), boundary labels (Qwen3Guard), heuristic word labels (FineHarm), or Monte Carlo rollouts (StreamGuard). Debounced triggers (Delay-k).
- **Future-latent objectives.** InfoNCE (CPC), SmoothL1 to stop-gradient future latents (NextLat, HiLP), multi-token heads (MTP), and suffix encoders used only at training time (BST).
- **Survival heads.** Discrete-time logistic hazard with censoring (Nnet-survival), Cox-Time, DeepHit.

## 4. Standard baselines (for our comparison)

1. **Direct activation verbalizer trained on the same labels.**
   - (a) LatentQA-style LoRA decoder (`code/latentqa`). The official Llama-3-8B decoder exists; for Qwen3, use repo support for `Qwen/Qwen3-4B-Instruct-2507` or the AO checkpoints.
   - (b) An Activation Oracle fine-tuned on our forecasting QA. Its yes/no answers or bin choices are scored by logit, following the AO-calibration finding.
2. **Linear / MLP probe** on the current position (and last-k mean-pooled), with one head per horizon bin. This is the "memoryless direct" baseline.
3. **Per-prefix value head** (StreamGuard/SCM style), trained on the same event labels.
4. **Text-only baseline.** A small LM or classifier reading the generated text prefix. This checks whether activations add anything over text (the warning from Building Better AOs).
5. **Ablations of the observer.** −memory (single position); −multi-scale latent loss; hazard head vs independent per-horizon binary heads.

## 5. Evaluation metrics

- **Lead-stratified discrimination.** For each anchor position t and lead bin [k₁,k₂) ∈ {[1,64), [64,128), [128,256), [256,512), [512,1024), [1024,2048)}: AUROC / AUPRC for "event occurs in (t+k₁, t+k₂]". The key plot is Δ(observer − verbalizer) vs lead, and the hypothesis predicts this gap grows.
- **Survival metrics.** Time-dependent concordance C^td; IPCW integrated Brier score and integrated negative binomial log-likelihood (pycox `EvalSurv`); calibration (ECE of S(t) at fixed horizons). Report both IPCW and "fully-followed subset" numbers, because censoring by response end is informative.
- **Streaming-monitor metrics.** OnTime / Late / Miss relative to the annotated onset (StreamGuard), tokens-before-onset at first trigger, false-positive rate on benign or XSTest responses, leak rate.
- **Long-form content forecasting.**
  - Cosine / retrieval accuracy (R@k among in-batch candidates) between predicted and actual future-window embeddings, using `sentence-transformers/all-MiniLM-L6-v2` (cached).
  - Topic/section-label F1, or QA accuracy judged by an LLM, per lead bin.
- **Length / EOS.** Spearman and MAE for remaining length (ForeLen).

## 6. Datasets in the literature (and what we have)

| Need | Dataset | Status |
|---|---|---|
| Token-level unsafe onset (short responses) | Qwen3GuardTest `response_loc` (813) | downloaded |
| Unsafe onset inside reasoning traces | Qwen3GuardTest `thinking_loc` (569), `thinking` (1,059) | downloaded |
| Harmful prompts for on-policy rollouts | HarmBench (CSV), StrongREJECT (CSV), AdvBench (`mlabonne/harmful_behaviors`), JBB-Behaviors, BeaverTails (330k), Aegis 2.0, LLM-LAT harmful, circuit-breakers (harmful completions avg ≈370 tokens) | downloaded |
| Benign or over-refusal controls | XSTest, harmless_alpaca, Dolly-15k (LatentQA stimuli) | downloaded |
| Long-form outputs | LongWriter-6k (outputs: median ≈4.7k tokens, p10 ≈2.8k), ForeLen Qwen2.5-7B LongSeq (1,057) | downloaded |
| Moderation-labeled chat | ToxicChat, Aegis 2.0, BeaverTails | downloaded |
| Token-level harm (FineHarm), shallow-vs-deep safety data, WildGuardMix, WildJailbreak, HEx-PHI | — | gated, not accessible with the current token |

**Measured horizon statistics** (`datasets/horizon_stats.txt`, Qwen3 tokenizer):
- `response_loc` onset: median 7 tokens into the response (max 106).
- `thinking_loc` onset: median 47 tokens, p90 111, max 307, inside traces of median 1,085 tokens.
- 0% of onsets are at ≥512 tokens.

**Conclusion:** natural single-turn safety data cannot test leads of 512–2048 tokens. Long-lead evaluation needs constructed or long-trajectory data; see §8.

## 7. Gaps and opportunities

1. **No lead-distance analysis** for any activation-based monitor or verbalizer. This is the main contribution opportunity.
2. **Verbalizers read a static window.** None keeps memory across a growing trajectory, although the U-shaped probe accuracy and the answer flips (≈3 per trace) suggest memory matters.
3. **Future-latent objectives** (NextLat, HiLP, CPC) have only been used for pretraining LMs at horizons of ≤20 tokens, never as auxiliary objectives for an *external observer* at horizons of 256–2048 tokens.
4. **No verbalizer outputs a calibrated event-time distribution.** Survival heads with censoring are standard in medicine but absent here.
5. **Benchmarks.** No public benchmark gives harm onsets far into long responses.

## 8. Research directions: enumeration, scoring and pruning (direction budget = 3)

Each direction is scored 1–5 on evidence (E), relevance to the hypothesis (R), expected information gain (I) and feasibility on 1× A6000 48 GB with about 60 GB of free disk (F). Total out of 20.

| # | Direction | E | R | I | F | Total | Decision |
|---|---|---|---|---|---|---|---|
| D1 | **Long-horizon unsafe event-time forecasting on constructed long trajectories.** Late-harm multi-part long-form prompts plus reasoning-trace → final-answer onsets; on-policy Qwen3-8B (or abliterated) rollouts; token onsets labeled by Qwen3Guard-Stream. Observer (memory + multi-scale future latents + hazard head) vs LatentQA/AO-style verbalizer and probe on the same labels; lead-stratified AUROC and C^td. | 4 | 5 | 5 | 4 | **18** | **KEEP (primary)** |
| D2 | **Long-form content forecasting.** Predict what future windows (+128 … +2048 tokens) will cover, as future-window sentence embeddings and topic QA, on LongWriter-6k / ForeLen on-policy rollouts; observer vs verbalizer per lead bin. | 4 | 5 | 4 | 4 | **17** | **KEEP** |
| D3 | **Component attribution and free-label sanity task.** Ablate memory, multi-scale latent loss and hazard head (vs per-horizon binary heads) on the D1/D2 data, plus time-to-EOS / answer-lock-in hazards (ForeLen, reasoning traces) as cheap labels that need no judge. | 4 | 4 | 4 | 5 | **17** | **KEEP** |
| D4 | Train new full-size AO/LatentQA decoders from scratch with hazard verbalization (many GPU-hours) | 3 | 4 | 3 | 1 | 11 | reject: infeasible compute; use LoRA fine-tuning of existing checkpoints as the baseline only |
| D5 | Multi-turn WildChat / LMSYS toxicity forecasting (turn-level) | 2 | 3 | 3 | 2 | 10 | reject: off-policy activations (the original model's states are unavailable), noisy turn-level labels, LMSYS gated |
| D6 | RealToxicityPrompts continuation toxicity | 3 | 2 | 2 | 5 | 12 | reject: continuations ≈20 tokens, no long-horizon signal |
| D7 | Steering or early intervention using observer forecasts | 2 | 2 | 3 | 3 | 10 | reject: out of scope (hypothesis is about forecasting); possible follow-up |
| D8 | Conformal lower bounds on time-to-unsafe on top of the hazard head | 3 | 3 | 2 | 4 | 12 | reject as a separate direction; optional calibration add-on inside D1 |
| D9 | Cross-model / scale study (4B → 8B → 32B) | 3 | 3 | 3 | 2 | 11 | reject: compute; only if D1 is cheap |

Rejected directions are pruned and should not be reopened unless D1–D3 results invalidate the ranking (see STATE.md).

## 9. Recommendations for the experiment

**Target model.**
- `Qwen/Qwen3-8B` (cached). It shares the tokenizer with Qwen3Guard-Stream (token-aligned onset labels, no re-tokenization), and an AO checkpoint for it is cached.
- Use `Qwen/Qwen3-4B(-Instruct-2507)` if throughput is limiting; LatentQA's code already supports it.
- `huihui-ai/Qwen3-8B-abliterated` (cached) can generate trajectories that actually turn unsafe. Read activations from the *same* model that generated the trajectory (on-policy).
- Only GPU 2 of the 4 A6000s was free when checked. Plan for one GPU.

**Long-lead data construction** (D1):
- (a) **Late-harm multi-part prompts.** A LongWriter-style benign long task (e.g. a 1,500-word report on X) followed by a final harmful sub-request from HarmBench/StrongREJECT/AdvBench. Generate with the abliterated model and label per-token onset with Qwen3Guard-Stream-4B. Onsets land roughly 300–2000 tokens in.
- (b) **Reasoning traces** (thinking mode) on harmful prompts. Event = first unsafe token of the *final answer* after `</think>`, or answer lock-in. Leads ≈500–1500 tokens.
- (c) Benign long-form controls (LongWriter, Dolly) that never turn unsafe. These give censored trajectories and calibrate the false-positive rate.

**Labels.**
- Per anchor position t: event time T−t (bins up to 2048), with right-censoring at end-of-response or at the 2048 horizon.
- For verbalizer parity, phrase the same labels as QA: "Will the response become unsafe within the next N tokens?" and "In which window?". Score answers by logit over a closed set.

**Observer architecture** (to implement; no code exists):
- (a) A small causal transformer or GRU over a sequence of compressed activations (e.g. every 8–16 tokens, 1–3 mid/late layers, projected to 256–512 dimensions), which acts as the memory.
- (b) Auxiliary heads that predict pooled future-window latents at scales {64, 256, 1024, 2048} with InfoNCE or SmoothL1 against stop-gradient targets (CPC, NextLat/HiLP). The targets are mean-pooled target-model activations or MiniLM embeddings of the future text window.
- (c) A discrete-time logistic hazard head with censoring (Nnet-survival loss).

**Baselines** (same labels, same activations): §4 items 1–4. The key comparison is *observer vs direct verbalizer*, stratified by lead bin.

**Metrics:** §5. Primary: AUROC per lead bin and C^td. Secondary: integrated Brier score, OnTime/Late/Miss, R@k for content.

**Practical cautions.**
- **Storage.** Full activations are 4096-d bf16, i.e. 8 KB per token per layer. Store subsampled positions (stride 8–16) and/or PCA to 512 dimensions; about 60 GB of disk is free.
- **Leakage.** Split by *prompt* (and by harmful behavior ID); never split anchors from the same trajectory across train and test.
- **Censoring.** Censoring is informative (long responses are different), so also report on trajectories followed for the full horizon.
- **Text-only control.** Include a text-only baseline to show that activations add information.
