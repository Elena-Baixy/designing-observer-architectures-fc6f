"""Per-prompt-source breakdown of safety forecasts (seed-averaged risk), H=256 and H=512."""
import numpy as np, sys
from sklearn.metrics import roc_auc_score, brier_score_loss
out = []
for ev in ["caution", "unsafe"]:
    risk = np.load(f"results/risk_{ev}.npy", allow_pickle=True).item(); m = np.load(f"results/test_meta_{ev}.npy", allow_pickle=True).item()
    Y, fam = m["Y"], m["fam"]; src = np.array([f.split("_")[0] if not f.startswith("toxic") else "toxic_chat" for f in fam])
    out.append(f"\n### event={ev}: per-source AUROC / Brier (H=512, known-status still-safe test prefixes; n, positives)\n")
    out.append("| source | n | pos | " + " | ".join(risk) + " |"); out.append("|---" * (3 + len(risk)) + "|")
    h = 3
    for s in sorted(set(src)):
        k = (src == s) & (Y[:, h] >= 0); y = Y[k, h]
        cells = []
        for name, r in risk.items():
            p = r.mean(0)[k, h]
            cells.append((f"{roc_auc_score(y, p):.2f}/" if 0 < y.sum() < len(y) else "–/") + f"{brier_score_loss(y, p):.3f}" if len(y) else "–")
        out.append(f"| {s} | {k.sum()} | {int(y.sum())} | " + " | ".join(cells) + " |")
open("results/per_family.md", "w").write("\n".join(out)); print("\n".join(out))
