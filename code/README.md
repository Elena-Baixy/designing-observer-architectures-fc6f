# Cloned Repositories

All cloned into `code/`. Treat them as third-party code. Our own helper scripts live in `scripts/`; run them from the workspace root with `PYTHONPATH=code/<repo>`.

## 1. LatentQA (USER-SPECIFIED, required)
- **URL:** https://github.com/aypan17/latentqa (full clone; HEAD `a2dcb6f`, "add changes to enable qwen and gemma models")
- **Location:** `code/latentqa/`
- **Purpose:** training and inference for the LatentQA decoder (LIT), the canonical *direct activation verbalizer* baseline.
- **Key files:**
  - `lit/train.py`: torchrun DDP/FSDP training. Args: `--target_model_name`, `--train_{stimulus_completion,stimulus,control,qa}`, `--min_layer_to_read 15 --max_layer_to_read 16 --layer_to_write 0`.
  - `lit/reading.py`: `interpret(target, decoder, tok, dialogs, questions, args)` generates answers.
  - `lit/control.py`: steering through decoder gradients.
  - `lit/utils/activation_utils.py`: `latent_qa(...)`, forward hooks that cache layer outputs from the target, and `generate_substitute_layer_single(...)`, which substitutes them at placeholder positions in the decoder.
  - `lit/utils/dataset_utils.py`: `lqa_tokenize`, `get_dataset`, `BASE_DIALOG`, and per-model magic-number tables (`NUM_READ/WRITE_TOKENS_TO_SHIFT`, `PAD_TOKEN_IDS`, `CHAT_FORMAT_TOKENS`, `ENCODER_CHAT_TEMPLATES`).
  - Supported models: Llama-3/3.1-8B, Llama-3-70B, Ministral-8B, Mistral-Small-24B, R1-Distill-Llama-8B/Qwen-32B, gemma-3-4b-it, Qwen3-4B-Instruct-2507.
  - `lit/configs/train_config.py`: defaults are lr 3e-5, batch 4, 5 epochs, LoRA r16/α32 (the paper used r32/α64, lr 1e-4, batch 128).
  - `data/train/`: control.json (16,732), stimulus.json / stimulus_completion.json / system.json (7,844), qa.json (dict, 16,784 keys). `data/eval/`: eval split.
- **Released decoder:** `aypan17/latentqa_llama-3-8b-instruct` (cached in `$HF_HOME=/data/smallyan/hf_home`).
- **Tested** with `scripts/test_latentqa_reading.py` on 1× A6000. The run works: both 8B models fit on one 48 GB GPU and load in about 10 s. Output is in `logs/latentqa_smoke_test.log`.
  - Example: from user-prompt activations alone, it answers "The assistant will adopt the persona of a pirate who secretly wants to steal the user's bank password."
  - For a meth-synthesis prompt it answers "...will cover the steps involved in synthesizing methamphetamine".
  - **It has no notion of *when* harm will occur.**
- **Compatibility issues found and handled in the wrapper; the repo itself is not modified:**
  1. `lit/reading.py` and `lit/control.py` import `tokenize`, which was renamed to `lqa_tokenize`. Fix: alias `du.tokenize = du.lqa_tokenize`.
  2. With transformers ≥5, `BatchEncoding[int]` returns a `tokenizers.Encoding`, so `reading.py` line 140 crashes. Fix: a shim makes `tokenized_write[i]` return `input_ids[i]`.
  3. `get_model` requests `flash_attention_2`, which is not installed. Fix: monkeypatch `from_pretrained` to use `attn_implementation="sdpa"`.
  4. The model-name keys are `meta-llama/Meta-Llama-3-8B-Instruct`; only `NousResearch/Meta-Llama-3-8B-Instruct` is cached (identical weights and tokenizer). Fix: alias the dict entries.
  - The repo pins `peft==0.10.0` and `transformers>=4.46.2`; the workspace venv has transformers 5.20.1, which works with the shims above.
- **How to adapt for our research:**
  - Train a LIT decoder on *our* forecasting QA ("Will the response become unsafe within the next N tokens?", "What will tokens t+256…t+512 cover?") using activations of the trajectory prefix. This gives the "direct verbalizer trained on the same labels" baseline.
  - For Qwen3, the closest supported model is `Qwen/Qwen3-4B-Instruct-2507`. Adding `Qwen/Qwen3-8B` needs new magic-number entries; copy the Qwen3-4B values, since the tokenizer is the same, then verify.

## 2. Activation Oracles (paper code)
- **URL:** https://github.com/adamkarvonen/activation_oracles (depth-1 clone of `main`; the `paper` branch / `v1.0-paper` tag reproduces the paper)
- **Location:** `code/activation_oracles/`
- **Purpose:** the general-purpose verbalizer baseline.
- **Key files:**
  - `nl_probes/sft.py`: training with torchrun; config in `nl_probes/configs/sft_config.py`.
  - `nl_probes/utils/steering_hooks.py`: norm-matched additive injection, `normalize(v)·||h||·coef`.
  - `nl_probes/dataset_classes/`: LatentQA, classification and context-prediction tasks.
  - `experiments/linear_probe.py`: probe baseline.
  - `experiments/activation_oracle_demo.ipynb`: self-contained inference.
- **Checkpoints:**
  - `adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B` (**cached**).
  - Collection https://huggingface.co/collections/adamkarvonen/activation-oracles also covers Qwen3-1.7B/4B/14B/32B, Gemma-2/3 and Llama-3.x.
- **Notes:** uses its own `uv sync` environment; in the paper version, injection happens after layer 1. Smaller oracles (1–4B) generalize worse out of distribution.

## 3. Building Better Activation Oracles
- **URL:** https://github.com/japhba/activation_oracles (depth-1 clone)
- **Location:** `code/building_better_activation_oracles/`
- **Purpose:** improved AO recipe and AObench.
  - On-policy rollouts.
  - CoT-chunk conversational QA asking about the *future* part of a reasoning trace.
  - 5-layer reading.
- **Key files:** `training_configs/*.json`, `nl_probes/sft.py --config ... --gen-only` (dataset generation via vLLM), `validation_suites/` (AObench).
- **Checkpoint:** `ceselder/qwen3-8b-ao-v3-best`. Dataset: `ceselder/cot-oracle-convqa-chunked-sonnet` (104k rows, ≈626 MB, not downloaded).
- **Notes:** needs vLLM for data generation.

## 4. NextLat (next-latent prediction)
- **URL:** https://github.com/JaydenTeoh/NextLat
- **Location:** `code/nextlat/`
- **Purpose:** reference implementations of auxiliary future-latent objectives: `models/model_nextlat.py` (NextLat), `model_mtp_gloeckle.py` (MTP), `model_mtp_jtp.py`, `model_bst.py` (Belief State Transformer), `model_probe.py`.
- **Use:** borrow the `compute_loss()` designs (SmoothL1 to stop-gradient future latents; BST suffix encoder) for the observer's multi-scale future-window heads.
- **Requirements:** PyTorch ≥2.6.

## 5. Qwen3Guard
- **URL:** https://github.com/QwenLM/Qwen3Guard
- **Location:** `code/qwen3guard/`
- **Purpose:** usage and evaluation of **Qwen3Guard-Stream**, a token-level safe/controversial/unsafe streaming classifier sharing the Qwen3 tokenizer. It is our automatic **event-time labeler**.
- **Key files:** `README.md` (streaming usage: `AutoModel.from_pretrained(..., trust_remote_code=True)`, then `model.stream_moderate_from_ids(...)` fed incrementally), `eval/eval_stream.py` (OnTime-style streaming evaluation on Qwen3GuardTest), `Qwen3Guard_Technical_Report.pdf`.
- **Models cached:** `Qwen/Qwen3Guard-Stream-0.6B`, `Qwen/Qwen3Guard-Stream-4B`.

## 6. pycox
- **URL:** https://github.com/havakv/pycox
- **Location:** `code/pycox/`
- **Purpose:** survival losses and metrics.
  - Losses: `models/logistic_hazard.py` (discrete-time hazard, the Nnet-survival-style NLL), `models/deephit.py`, `models/cox_time.py`, `models/loss.py`.
  - Metrics: `evaluation/` (`EvalSurv`: time-dependent concordance, IPCW Brier score / integrated Brier score / integrated negative binomial log-likelihood).
- **Notes:** can be installed with `uv pip install pycox`, or the loss and metric code can simply be copied (pure PyTorch/numpy).

## Locally cached models relevant to the experiments (`$HF_HOME/hub`)

| Model | Use |
|---|---|
| `Qwen/Qwen3-8B` | Recommended target model (on-policy rollouts plus activations). |
| `huihui-ai/Qwen3-8B-abliterated`, `huihui-ai/Huihui-Qwen3-8B-abliterated-v2` | Generate trajectories that actually become unsafe; read their own activations. |
| `adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B` | AO verbalizer baseline for Qwen3-8B. |
| `NousResearch/Meta-Llama-3-8B-Instruct` + `aypan17/latentqa_llama-3-8b-instruct` | Official LatentQA target and decoder pair. |
| `Qwen/Qwen3Guard-Stream-0.6B`, `-4B` | Token-level unsafe-onset labeler. |
| `sentence-transformers/all-MiniLM-L6-v2` | Embedding targets for future-window content (D2). |
