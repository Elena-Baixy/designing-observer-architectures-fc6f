# Observer architectures for long-horizon forecasting (frozen Qwen3-8B)

This project trains small "observers" that read a frozen Qwen3-8B's activations while it generates. They forecast two things:
1. When the response will first turn unsafe (or start refusing/warning) within H = 64…2048 tokens.
2. What the remaining response will do: per-segment behavior and topic, plus a summary embedding.

Three new architectures were compared with a parameter-matched direct activation readout ("direct_AO"):
- A, horizon-query bottleneck;
- B, latent rollout;
- D, hierarchical coarse→fine.

All three include streaming memory and a JEPA future-activation loss, and all use a censoring-aware hazard head. Ablations and zero/shuffle controls were also run. This is a pilot: 96 prompts, 84 usable trajectories, and a held-out split by prompt family. Full details are in **[REPORT.md](REPORT.md)**.

## Key findings
- **Unsafe-onset forecasting is unpowered.** On unmodified benchmark prompts, Qwen3-8B produced only 5 unsafe onsets in 84 trajectories, all ≤448 tokens. No architecture differs from direct_AO (all CIs include 0), and text-only TF-IDF is the best unsafe predictor.
- **Caution onsets (first refuse/warn):** hierarchical (D) and horizon-query (A) observers rank better than direct_AO by +0.03 to +0.11 AUROC at H ≤ 1024. This is not significant (best: D at H=512, p = 0.09). The gain peaks at 256–512 and does **not** grow with lead distance. Pooled AUROC is largely driven by separating prompt sources.
- **Memory over the activation trajectory** is the component with the clearest long-H benefit. Removing it from the rollout model costs about 0.1 AUROC at H ≥ 512.
- **No measurable benefit** from the JEPA auxiliary, the hazard head, or latent rollout. The hazard head is worse calibrated at long H under heavy censoring.
- **Long-form content:** observers forecast future-segment behavior at 0.85–0.88 accuracy up to 1024 tokens ahead (majority baseline 0.62, copy-last-segment 0.82–0.84). Remaining-summary retrieval reaches MRR ≈ 0.10 vs 0.019 for the no-activation baseline. A and D beat direct_AO on summary retrieval; direct_AO is best on topic. Zero/shuffle controls collapse performance, so the observers do read activations.

## Reproduce
```bash
uv venv .venv && uv sync   # or: uv pip install -r pyproject deps (torch, transformers, sklearn, openai, matplotlib)
export HF_HOME=/path/to/hf_cache CUDA_VISIBLE_DEVICES=0 OPENAI_API_KEY=...
N_PER=16 BS=48 MAXNEW=2048 .venv/bin/python src/gen_and_cache.py          # prompts+split, generation, activation cache (~10 min, A6000)
.venv/bin/python src/annotate.py artifacts/generations.jsonl artifacts/annotations.jsonl   # LLM timeline annotation (gpt-5.6-luna)
EVENT=unsafe  .venv/bin/python src/train_observers.py                      # baselines + 8 observer configs x 3 seeds (~2 min)
EVENT=caution .venv/bin/python src/train_observers.py
.venv/bin/python src/analyze.py unsafe; .venv/bin/python src/analyze.py caution; .venv/bin/python src/per_family.py
```

## Files
- `src/gen_and_cache.py`: builds prompts, splits by family before generation, runs Qwen3-8B generation, caches layer 9/18/27/35 16-token block activations.
- `src/annotate.py`: one LLM annotation per generation, giving a 64-token-segment timeline.
- `src/train_observers.py`: targets, leakage/split audits, baselines, architectures A/B/D, direct_AO, ablations, controls, metrics.
- `src/analyze.py`, `src/per_family.py`: tables, paired bootstrap tests, figures.
- `results/`: `results_{event}.json` (all metrics), `tables_{event}.md`, `per_family.md`, `data_stats.json`, `example_samples.json`, `risk_*.npy`.
- `figures/`: AUROC vs H and Δ vs direct_AO; content accuracy vs distance.
- `artifacts/`: generations, annotations, activation cache (not in git).
- `planning.md`, `literature_review.md`, `resources.md`.
