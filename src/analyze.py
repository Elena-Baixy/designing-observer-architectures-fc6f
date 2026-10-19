"""Aggregate results across seeds, paired significance tests vs direct_AO, figures and markdown tables."""
import json, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

EVENT = sys.argv[1] if len(sys.argv) > 1 else "unsafe"
import os; RD = os.environ.get("RES", "results"); FD = os.environ.get("FIG", "figures")
R = json.load(open(f"{RD}/results_{EVENT}.json"))
HS = ["64", "128", "256", "512", "1024", "2048"]
risk = np.load(f"{RD}/risk_{EVENT}.npy", allow_pickle=True).item()
meta = np.load(f"{RD}/test_meta_{EVENT}.npy", allow_pickle=True).item()
Y, fam = meta["Y"], meta["fam"]
models = list(risk.keys())
baselines = ["prevalence", "position", "linear_probe", "text_only_tfidf"]
out = []


def seeds_of(name, key, H, src="safety"):
    v = [R[src][k][H].get(key) for k in R[src] if k.startswith(name + "/seed")]
    return [x for x in v if x is not None]


def fmt(v): return "–" if not v else (f"{np.mean(v):.3f}±{np.std(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}")


out.append(f"## Safety forecasting (event={EVENT}); test still-safe prefixes\n")
st = R["stats"]
out.append(f"Known-status test prefixes per H: {st['test_known_per_H']}; positives: {st['test_pos_per_H']}\n")
for key in ["auroc", "brier", "auprc"]:
    out.append(f"\n**{key.upper()}** (mean±sd over seeds; strict censoring)\n")
    out.append("| method | " + " | ".join(f"H={h}" for h in HS) + " |"); out.append("|---" * (len(HS) + 1) + "|")
    for b in baselines:
        out.append(f"| {b} | " + " | ".join(fmt([R['safety'][b][h].get(key)] if R['safety'][b][h].get(key) is not None else []) for h in HS) + " |")
    for m in models:
        out.append(f"| {m} | " + " | ".join(fmt(seeds_of(m, key, h)) for h in HS) + " |")

# paired test vs direct_AO: seed-averaged risk, family-cluster bootstrap of AUROC difference
out.append("\n**AUROC difference vs direct_AO** (seed-averaged risk; 95% family-cluster bootstrap CI; p = two-sided bootstrap)\n")
out.append("| method | " + " | ".join(f"H={h}" for h in HS) + " |"); out.append("|---" * (len(HS) + 1) + "|")
rng = np.random.default_rng(0); uf = np.unique(fam)
boots = [np.concatenate([np.where(fam == f)[0] for f in rng.choice(uf, len(uf))]) for _ in range(500)]
base = risk["direct_AO"].mean(0)
diffs = {}
for m in models:
    if m == "direct_AO": continue
    pm = risk[m].mean(0); cells = []
    for h in range(len(HS)):
        k = Y[:, h] >= 0; y = Y[:, h]
        if not (0 < y[k].sum() < k.sum()): cells.append("–"); continue
        d0 = roc_auc_score(y[k], pm[k, h]) - roc_auc_score(y[k], base[k, h]); ds = []
        for b in boots:
            bb = b[k[b]]
            if 0 < y[bb].sum() < len(bb): ds.append(roc_auc_score(y[bb], pm[bb, h]) - roc_auc_score(y[bb], base[bb, h]))
        lo, hi = np.percentile(ds, [2.5, 97.5]); p = 2 * min(np.mean(np.array(ds) <= 0), np.mean(np.array(ds) >= 0))
        diffs.setdefault(m, []).append(d0); cells.append(f"{d0:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.2f}")
    out.append(f"| {m} | " + " | ".join(cells) + " |")

out.append("\n**Early-warning recall at 5% trajectory false-alarm rate (F(512) alarm)**, first seed\n")
out.append("| method | lead≥128 | lead≥256 | lead≥512 | lead≥1024 |"); out.append("|---|---|---|---|---|")
for k, v in R["ew"].items():
    if "/ctrl" in k or (k.count("/seed") and not k.endswith("seed0")): continue
    if v: out.append(f"| {k} | " + " | ".join(f"{v[L]['recall']:.2f} (n={v[L]['n']})" if L in v else "–" for L in ["128", "256", "512", "1024"]) + " |")

out.append("\n**Controls (seed 0): AUROC with zero / shuffled activations at test time**\n")
out.append("| method | ctrl | " + " | ".join(f"H={h}" for h in HS) + " |"); out.append("|---|---" + "|---" * len(HS) + "|")
for m in models:
    for c in ["zero", "shuffle"]:
        r = R["safety"].get(f"{m}/ctrl_{c}")
        if r: out.append(f"| {m} | {c} | " + " | ".join(f"{r[h].get('auroc', float('nan')):.3f}" for h in HS) + " |")

# ---------------- content
out.append(f"\n## Long-form content forecasting (all test cutoffs)\n")
C = R["content"]
names = ["majority", "copy_last_segment(oracle-annot)"] + models
out.append("| method | beh acc k=1 | k=4 | k=8 | k=16 | topic acc k=1 | k=4 | k=8 | k=16 | summary MRR | summary top10 |")
out.append("|---" * 11 + "|")
def agg(name, key, idx=None):
    ks = [k for k in C if k == name or k.startswith(name + "/seed")]
    vals = [C[k][key][idx] if idx is not None else C[k][key] for k in ks]
    vals = [v for v in vals if v == v]
    return fmt(vals)
for n in names:
    out.append(f"| {n} | " + " | ".join(agg(n, "beh_acc", i) for i in [0, 3, 7, 15]) + " | "
               + " | ".join(agg(n, "top_acc", i) for i in [0, 3, 7, 15]) + f" | {agg(n, 'summ_mrr')} | {agg(n, 'summ_top10')} |")
out.append("\nControls (content, seed 0): " + "; ".join(f"{k}: beh@k=4 {C[k]['beh_acc'][3]:.3f}, MRR {C[k]['summ_mrr']:.3f}" for k in C if "/ctrl" in k))
out.append("\n## Cost\n\n| method | params | train s (1 seed) | latency ms/update |\n|---|---|---|---|")
for k, v in R["cost"].items(): out.append(f"| {k} | {v['params']:,} | {v['train_s']:.0f} | {v['latency_ms']:.1f} |")
open(f"{RD}/tables_{EVENT}.md", "w").write("\n".join(out)); print("\n".join(out))

# ---------------- figures
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
x = np.arange(len(HS))
for b in baselines:
    ax[0].plot(x, [R["safety"][b][h].get("auroc", np.nan) for h in HS], "--", alpha=.6, label=b)
for m in models:
    if "no_" in m: continue
    v = [np.mean(seeds_of(m, "auroc", h)) if seeds_of(m, "auroc", h) else np.nan for h in HS]
    e = [np.std(seeds_of(m, "auroc", h)) if seeds_of(m, "auroc", h) else 0 for h in HS]
    ax[0].errorbar(x, v, yerr=e, marker="o", capsize=3, label=m)
ax[0].axhline(.5, color="k", lw=.5); ax[0].set_xticks(x, HS); ax[0].set_xlabel("horizon H (tokens)"); ax[0].set_ylabel("AUROC (test families)")
ax[0].set_title(f"Onset forecasting: {EVENT}"); ax[0].legend(fontsize=7)
for m in models:
    if m in ("direct_AO",) or "no_" in m or True:
        v = [np.mean(seeds_of(m, "auroc", h)) - np.mean(seeds_of("direct_AO", "auroc", h)) if seeds_of(m, "auroc", h) else np.nan for h in HS]
        if m != "direct_AO": ax[1].plot(x, v, marker="o", label=m)
ax[1].axhline(0, color="k", lw=.5); ax[1].set_xticks(x, HS); ax[1].set_xlabel("horizon H (tokens)"); ax[1].set_ylabel("ΔAUROC vs direct_AO")
ax[1].set_title("Gain vs lead distance (incl. ablations)"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{FD}/safety_auroc_vs_H_{EVENT}.png", dpi=130)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for n in names:
    if "no_" in n: continue
    for j, key in enumerate(["beh_acc", "top_acc"]):
        ks = [k for k in C if k == n or k.startswith(n + "/seed")]
        ax[j].plot(np.arange(1, 17) * 64, np.nanmean([C[k][key] for k in ks], 0), marker=".", label=n)
for j, t in enumerate(["future-segment behavior accuracy", "future-segment topic-cluster accuracy"]):
    ax[j].set_xlabel("distance ahead (tokens)"); ax[j].set_title(t); ax[j].set_xscale("log", base=2)
ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{FD}/content_vs_distance_{EVENT}.png", dpi=130)
print("figures saved")
