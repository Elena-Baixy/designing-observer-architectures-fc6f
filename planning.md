# Planning: observer architectures for long-horizon forecasting (Qwen3-8B, frozen)

## Motivation & Novelty Assessment

### Why This Research Matters
Runtime monitors that only flag harm once it has appeared react too late for long agentic or long-form generations. An observer that reads a frozen model's activations and forecasts *when* a response will turn unsafe, and *what* the rest of a long answer will do, would enable early intervention and give insight into how far ahead the model "plans".

### Gap in Existing Work
literature_review.md finds:
- No prior work evaluates activation-based forecasting by lead distance beyond about 32 tokens. Probes, Future Lens, and next-latent prediction all stay at ≤32 tokens.
- LatentQA and Activation Oracles read one activation and answer about properties of the completion, but they have no time axis, no trajectory memory, and no event-time output.
- Safety forecasting work such as StreamGuard and "time to unsafe sampling" either works on text or does not model censoring.

### Our Novel Contribution
We run a matched comparison of new observer designs against a matched *direct activation readout* on the same labels, with horizons to 2048 tokens:
- A, horizon-query bottleneck;
- B, latent rollout;
- C, a JEPA future-embedding auxiliary;
- D, hierarchical coarse→fine;
- E, streaming memory.

All designs share a censoring-aware discrete-time hazard head. Results are reported separately for safety-onset and long-form content forecasting, with family-cluster bootstrap CIs.

### Experiment Justification
- Exp 1 (data + audit): on-policy Qwen3-8B continuations, LLM timeline annotation, fixed 64-token cutoffs. Every later claim depends on leak-free, family-split targets.
- Exp 2 (baselines): prevalence (KM), position, text-only TF-IDF, linear probe, majority / copy-last content. These establish whether there is any activation signal beyond trivial cues.
- Exp 3 (architectures vs direct AO, 3 seeds): the core hypothesis test.
- Exp 4 (ablations: no memory, no JEPA, no hazard head; B without memory): attributes any gain to a component.
- Exp 5 (controls: zero / shuffled activations at test time): checks that the observers actually read activations.

## Research Question
Does an observer with (a) trajectory memory, (b) multi-window future-latent prediction and (c) an explicit hazard distribution beat a matched direct activation readout? The forecast targets are (1) first unsafe onset within H ∈ {64,…,2048} tokens and (2) future-segment behavior/topic and a remaining-response summary. Does the gain grow with lead distance?

## Hypothesis Decomposition
- H1: the new architectures have higher AUROC and lower Brier than direct_AO, averaged over H.
- H2: the gap grows with H.
- H3: each of memory, the JEPA auxiliary, and the hazard head contributes. Removing one hurts at long H.
- H4: content accuracy is above majority and copy-last baselines at distance ≥ 4 segments (≥256 tokens).

## Design decisions (fixed before looking at continuations)
- **Prompt sources (unmodified benchmark prompts, no adversarial wrappers).** ToxicChat (toxic real user prompts), BeaverTails (unsafe-category prompts), Aegis 2.0 (unsafe prompts), XSTest (safe/contrast prompts) and LongWriter-6k (long-form). 40 prompts per source, 216 prompts total.
  - Families = source × category (or a LongWriter bucket). A 30% family-level test split was assigned *before generation*.
  - *Deviation:* an earlier idea of wrapping harmful requests in long-form templates to construct late onsets was dropped. The study therefore measures naturally occurring events only, and these are expected to be rare and early.
- **Generation.** Qwen3-8B, non-thinking mode, T=0.7, top-p 0.9, max 2048 new tokens.
- **Events.**
  - Primary: the first 64-token segment the annotator marks unsafe.
  - Secondary, denser event ("caution"): the first segment with behavior refuse or warn. It is used because natural unsafe events may be too rare.
- **Censoring.** Generations that stop (EOS or cap) without the event are right-censored, not safe. A secondary evaluation treats EOS-without-event as event-free.
- **Cutoffs.** Every 64 tokens. The observer reads 16-token block-mean activations of layers 9/18/27/35 for blocks up to the cutoff only; this is asserted in code. Only still-safe prefixes are used for safety.
- **Annotator.** gpt-5.6-luna, called directly through the OpenAI API because no OpenRouter key is available in this environment. One call per generation.

## Baselines and metrics
- Baselines are listed above. Direct_AO = an MLP on the latest activation with independent per-horizon sigmoids and per-window content heads. It has the same encoder, epochs and targets as the observers. It is a cheap stand-in for a trained AO/LatentQA verbalizer: a full LLM-decoder AO was infeasible in the 1-hour budget.
- Safety metrics: AUROC, AUPRC, Brier and log loss per H; early-warning recall at a 5% trajectory false-alarm rate.
- Content metrics: behavior and topic accuracy vs distance; remaining-summary retrieval (MRR, cosine).
- Cost: parameters, training seconds, latency.
- Statistics: family-cluster bootstrap 95% CIs; 3 seeds.

## Timeline (60 min)
- Generation and caching: about 20 min.
- Annotation: about 3 min.
- Training: about 10 min.
- Analysis and docs: about 15 min.

## Potential Challenges
- Few unsafe events. Mitigation: the caution event, and honest reporting of power.
- GPU memory contention.
- Annotation noise.
- Short responses: censoring leaves few known labels at long H.

## Direction ranking (direction budget = 3)

Scored on evidence, relevance, information gain and feasibility; see also literature_review.md §8.

**Kept:**
1. Long-horizon onset forecasting with censoring-aware hazard observers vs a matched direct readout (D1).
2. Long-form content forecasting: future-segment behavior/topic and remaining-summary retrieval (D2).
3. Component ablations: memory, JEPA future-latent loss, hazard head, latent rollout; plus zero/shuffle controls (D3).

**Rejected:**
- Training new full LLM-decoder AOs from scratch: compute and time.
- Multi-turn WildChat: gated and off-policy.
- RealToxicityPrompts: horizon too short.
- Steering: out of scope.
- Conformal bounds: optional.
- Scale study: compute.
- *Constructed* late-harm prompts that would elicit detailed harmful content: dropped during execution on safety grounds. Only naturally occurring events are used.
