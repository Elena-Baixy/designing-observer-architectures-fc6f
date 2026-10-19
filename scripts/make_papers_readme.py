import json, glob, os
meta = {m["id"]: m for m in json.load(open("papers/arxiv_meta.json"))}
why = {
"2412.08686":"USER-SPECIFIED. LatentQA/LIT decoder: canonical direct activation verbalizer baseline; code in code/latentqa.",
"2512.15674":"Activation Oracles: general-purpose verbalizer baseline; released Qwen3/Gemma/Llama checkpoints.",
"2606.02609":"Better AOs: on-policy + future-CoT QA, multi-layer reading; AObench; logit-AUROC scoring advice.",
"2605.26045":"AO calibration: candidate scoring gives calibrated closed-set answers -> fair hazard-bin scoring of verbalizers.",
"2609.34033":"Faithful activation verbalization (AVPO): hallucination failure mode of verbalizers.",
"2512.15712":"Predictive Concept Decoders: future-token pretraining improves jailbreak/harm forecasting vs LatentQA.",
"2401.06102":"Patchscopes: untrained patching verbalizer baseline.",
"2403.10949":"SelfIE: untrained self-interpretation baseline.",
"2410.02472":"Meta-models: decoding behaviors from activations.",
"2311.04897":"Future Lens: decay of future-token predictability from one hidden state.",
"2404.00859":"Do LMs plan ahead: pre-caching vs breadcrumbs.",
"2502.06258":"Emergent response planning: prompt states encode global future attributes; U-shaped accuracy over response.",
"2604.22266":"Decide early explain later: answer locked ~760-1900 tokens before end; natural long-lead event.",
"2602.11812":"Length prediction from hidden states (EGTP/PLP); ForeLen dataset; time-to-EOS labels.",
"2604.03962":"StreamGuard: safety moderation as forecasting via MC rollouts; OnTime/Late/Miss protocol.",
"2506.09996":"SCM/FineHarm: token-level harm labels and streaming early stopping.",
"2406.05946":"Shallow safety alignment: harm decided in first few tokens; prefill attacks, recovery.",
"2406.11717":"Refusal direction: linear refusal features for probe baselines.",
"2501.18837":"Constitutional Classifiers: streaming output classifier design + leak-rate metrics.",
"2506.13593":"Time-to-unsafe-sampling: survival framing + conformal bounds for unsafe events.",
"2502.16797":"Forecasting rare LM behaviors at deployment scale.",
"2407.03282":"Hallucination risk from query internal states (pre-generation probe).",
"2502.03407":"Linear probes for strategic deception (probe-monitor baseline practices).",
"1807.03748":"CPC: InfoNCE multi-horizon future latent prediction.",
"2404.19737":"Multi-token prediction: horizon limits of token-space future objectives.",
"2410.23506":"Belief State Transformer: suffix encoder, belief states.",
"2511.05963":"NextLat: next-latent prediction -> belief states (code in code/nextlat).",
"2608.05806":"HiLP: hierarchical (window-pooled) latent prediction; gain grows with horizon.",
"1805.00917":"Nnet-survival: discrete-time hazard loss with censoring (observer hazard head).",
"1907.00825":"Cox-Time / pycox: survival losses and C^td / IBS metrics (code/pycox).",
"2510.14276":"Qwen3Guard: Qwen3Guard-Stream token-level safety labeler; Qwen3GuardTest benchmark.",
}
extra = {"2510.14276": dict(title="Qwen3Guard Technical Report", authors=["Qwen Team"], year=2025)}
lines = ["# Downloaded Papers", "", f"{len(glob.glob('papers/*.pdf'))} PDFs. Per-paper notes: `papers/notes/<arxiv_id>.md`. Page chunks: `papers/pages/`.", ""]
for i, f in enumerate(sorted(glob.glob("papers/*.pdf")), 1):
    aid = os.path.basename(f).split("_")[0]
    m = meta.get(aid, extra.get(aid, {}))
    au = ", ".join(m.get("authors", [])[:4]) + (" et al." if len(m.get("authors", [])) > 4 else "")
    note = f" | notes: notes/{aid}.md" if os.path.exists(f"papers/notes/{aid}.md") else ""
    lines += [f"{i}. [{m.get('title','?')}]({os.path.basename(f)})", f"   - Authors: {au}", f"   - Year: {m.get('year','?')} | arXiv: {aid}{note}", f"   - Why relevant: {why.get(aid,'')}", ""]
open("papers/README.md", "w").write("\n".join(lines))
print("\n".join(lines[:12]))
