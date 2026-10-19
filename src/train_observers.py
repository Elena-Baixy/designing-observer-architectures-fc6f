"""Steps 3-8: derive prefix-level targets, audit causality/splits, train baselines and observer
architectures on cached Qwen3-8B activations, evaluate safety (hazard) and long-form content forecasts.

Cutoffs are fixed a priori at every 64 generated tokens (t = 0, 64, 128, ...). At cutoff t the
observer reads activation blocks 0..t/16 only (block 0 = end of prompt; block j = tokens [16(j-1),16j)).
"""
import json, os, sys, time, math, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel

ART = os.environ.get("ART", "artifacts"); RES = os.environ.get("RES", "results")
SEG, BLK = 64, 16
R = SEG // BLK                  # blocks per segment
NB = 32                         # hazard bins (64 tokens each) -> 2048 tokens
HORIZONS = [64, 128, 256, 512, 1024, 2048]
KC = 16                         # content forecast distance (segments ahead)
BEH = ["explain", "instruct", "recommend", "narrate", "refuse", "warn", "other"]
NTOP = 16
W = 64                          # memory window in blocks (=1024 tokens)
D = 256
dev = "cuda"
EVENT = os.environ.get("EVENT", "unsafe")   # 'unsafe' or 'caution' (refuse/warn onset)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


# ---------------------------------------------------------------- data
def load():
    G = {json.loads(l)["id"]: json.loads(l) for l in open(f"{ART}/generations.jsonl")}
    A = {}
    for l in open(f"{ART}/annotations.jsonl"):
        a = json.loads(l)
        if "error" not in a: A[a["id"]] = a
    acts = np.load(f"{ART}/acts.npz")
    ids = sorted(i for i in A if str(i) in acts.files and len(G[i]["gen_ids"]) >= SEG)
    return G, A, acts, ids


def minilm(texts):
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    m = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(dev).eval()
    out = []
    for i in range(0, len(texts), 256):
        e = tok(texts[i:i + 256], padding=True, truncation=True, max_length=256, return_tensors="pt").to(dev)
        with torch.no_grad():
            h = m(**e).last_hidden_state
        msk = e["attention_mask"].unsqueeze(-1).float()
        out.append(F.normalize((h * msk).sum(1) / msk.sum(1), dim=-1).cpu())
    return torch.cat(out).numpy()


def build(G, A, acts, ids):
    """Build per-trajectory tensors and the list of (traj, cutoff c) samples with targets."""
    trajs, samples = [], []
    topics_all = []
    for i in ids:
        segs = A[i]["segments"]; K = len(segs)
        x = acts[str(i)].astype(np.float32)          # [nb+1, 4, d]
        if EVENT == "unsafe":
            ev = [k for k, s in enumerate(segs) if s["unsafe"]]
        else:
            ev = [k for k, s in enumerate(segs) if s["behavior"] in ("refuse", "warn")]
        e = ev[0] if ev else None
        trajs.append(dict(id=i, x=x, K=K, e=e, segs=segs, split=G[i]["split"], family=G[i]["family"],
                          src=G[i]["src"], eos=G[i]["ended_eos"], gen_ids=G[i]["gen_ids"], prompt=G[i]["prompt"]))
        topics_all += [s["topic"] or "none" for s in segs]
    return trajs, topics_all


def targets_for(tr, c, topic_lab, summ_emb):
    K, e = tr["K"], tr["e"]
    # hazard bins: k-th bin = segment c+k
    ev_bin = (e - c) if (e is not None and 0 <= e - c < NB) else -1
    n_obs = min(K - c, NB) if ev_bin < 0 else ev_bin + 1        # bins with known status
    beh = np.full(KC, -1); top = np.full(KC, -1)
    for k in range(KC):
        if c + k < K:
            beh[k] = BEH.index(tr["segs"][c + k]["behavior"]); top[k] = topic_lab[tr["id"]][c + k]
    return dict(ev_bin=ev_bin, n_obs=n_obs, beh=beh, top=top, summ=summ_emb[(tr["id"], c)])


# ---------------------------------------------------------------- models
class Enc(nn.Module):
    """Shared input projection of multi-layer block features (4 layers x 4096 -> D)."""
    def __init__(self, din):
        super().__init__()
        self.ln = nn.LayerNorm(din); self.proj = nn.Linear(din, D)

    def forward(self, x): return self.proj(self.ln(x))


class Heads(nn.Module):
    """Heads reading per-future-window forecast vectors f [B, NB, D]."""
    def __init__(self, jepa_dim):
        super().__init__()
        self.haz = nn.Linear(D, 1); self.beh = nn.Linear(D, len(BEH)); self.top = nn.Linear(D, NTOP)
        self.summ = nn.Linear(D, 384); self.jepa = nn.Linear(D, jepa_dim)

    def forward(self, f):
        return dict(haz_logit=self.haz(f).squeeze(-1), beh=self.beh(f[:, :KC]), top=self.top(f[:, :KC]),
                    summ=self.summ(f.mean(1)), jepa=self.jepa(f))


class DirectAO(nn.Module):
    """Matched direct readout: latest activation only, MLP, independent per-horizon / per-window heads
    (no memory, no hazard structure, no future-latent objective)."""
    def __init__(self, din, jepa_dim):
        super().__init__()
        self.enc = Enc(din)
        self.mlp = nn.Sequential(nn.GELU(), nn.Linear(D, 1024), nn.GELU(), nn.Linear(1024, 1024), nn.GELU(), nn.Linear(1024, D))  # widened to match observer params
        self.horiz = nn.Linear(D, len(HORIZONS))
        self.beh = nn.Linear(D, KC * len(BEH)); self.top = nn.Linear(D, KC * NTOP); self.summ = nn.Linear(D, 384)
        self.hazard = False

    def forward(self, xs, m):
        h = self.mlp(self.enc(xs[:, -1]))
        B = h.shape[0]
        return dict(cls_logit=self.horiz(h), beh=self.beh(h).view(B, KC, -1), top=self.top(h).view(B, KC, -1),
                    summ=self.summ(h))


class Memory(nn.Module):
    """Streaming memory: causal transformer over the most recent W blocks (16-token updates)."""
    def __init__(self, din, use_memory=True):
        super().__init__()
        self.enc = Enc(din); self.pos = nn.Parameter(torch.zeros(W, D)); self.use_memory = use_memory
        L = nn.TransformerEncoderLayer(D, 4, 2 * D, dropout=0.1, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(L, 2)

    def forward(self, xs, m):
        h = self.enc(xs) + self.pos
        if not self.use_memory:                       # ablation: only the latest block is visible
            m = torch.zeros_like(m); m[:, -1] = 1
        causal = torch.triu(torch.ones(W, W, dtype=torch.bool, device=xs.device), 1)
        eye = torch.eye(W, dtype=torch.bool, device=xs.device)
        # True = blocked: future keys or padded keys; every position may attend to itself (avoids NaN rows)
        blocked = (causal.unsqueeze(0) | (m == 0).unsqueeze(1)) & ~eye.unsqueeze(0)
        blocked = blocked.repeat_interleave(4, 0)
        return self.tf(h, mask=blocked), m


class HorizonQuery(nn.Module):
    """Arch A(+E,+C): learned queries per future 64-token window cross-attend to the memory."""
    def __init__(self, din, jepa_dim, use_memory=True, hazard=True):
        super().__init__()
        self.mem = Memory(din, use_memory); self.q = nn.Parameter(torch.randn(NB, D) * 0.02)
        self.xa = nn.MultiheadAttention(D, 4, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))
        self.heads = Heads(jepa_dim); self.hazard = hazard
        if not hazard: self.horiz = nn.Linear(D, len(HORIZONS))

    def forward(self, xs, m):
        h, m = self.mem(xs, m)
        q = self.q.unsqueeze(0).expand(h.shape[0], -1, -1)
        f, _ = self.xa(q, h, h, key_padding_mask=(m == 0)); f = f + q; f = f + self.ff(f)
        o = self.heads(f)
        if not self.hazard: o["cls_logit"] = self.horiz(f.mean(1))
        return o


class Rollout(nn.Module):
    """Arch B: latent world model; GRU dynamics rolls a latent state forward one 64-token segment
    per step without generating tokens; heads read each rolled-out state."""
    def __init__(self, din, jepa_dim, use_memory=True):
        super().__init__()
        self.mem = Memory(din, use_memory); self.cell = nn.GRUCell(D, D); self.step = nn.Parameter(torch.randn(NB, D) * 0.02)
        self.heads = Heads(jepa_dim); self.hazard = True

    def forward(self, xs, m):
        h, m = self.mem(xs, m)
        z = h[:, -1]; zs = []
        for k in range(NB):
            z = self.cell(self.step[k].expand_as(z), z); zs.append(z)
        return self.heads(torch.stack(zs, 1))


class Hier(nn.Module):
    """Arch D: multi-scale. A coarse vector for the whole remaining response (from memory), then
    fine window forecasts conditioned on coarse = queries + coarse, cross-attending memory."""
    def __init__(self, din, jepa_dim):
        super().__init__()
        self.mem = Memory(din); self.cq = nn.Parameter(torch.randn(1, D) * 0.02); self.q = nn.Parameter(torch.randn(NB, D) * 0.02)
        self.xa1 = nn.MultiheadAttention(D, 4, batch_first=True); self.xa2 = nn.MultiheadAttention(D, 4, batch_first=True)
        self.cond = nn.Linear(D, D); self.heads = Heads(jepa_dim); self.hazard = True

    def forward(self, xs, m):
        h, m = self.mem(xs, m); B = h.shape[0]
        c, _ = self.xa1(self.cq.expand(B, -1, -1), h, h, key_padding_mask=(m == 0))
        q = self.q.unsqueeze(0) + self.cond(c)
        f, _ = self.xa2(q, h, h, key_padding_mask=(m == 0)); f = f + q
        o = self.heads(f); o["summ"] = self.heads.summ(c.squeeze(1) + f.mean(1))
        return o


# ---------------------------------------------------------------- losses
def hazard_nll(logit, ev_bin, n_obs):
    """Discrete-time logistic-hazard NLL with right censoring (censored = unknown after n_obs bins)."""
    k = torch.arange(NB, device=logit.device).unsqueeze(0)
    at_risk = (k < n_obs.unsqueeze(1)).float()
    y = (k == ev_bin.unsqueeze(1)).float()
    l = F.binary_cross_entropy_with_logits(logit, y, reduction="none") * at_risk
    return l.sum(1).mean()


def cum_risk(logit):
    """F(H) = 1 - prod_{k < H/64} (1 - h_k): monotone in H by construction."""
    logS = torch.cumsum(F.logsigmoid(-logit), 1)
    return torch.stack([1 - logS[:, H // SEG - 1].exp() for H in HORIZONS], 1)


def horizon_labels(ev_bin, n_obs):
    """Per-horizon binary label; -1 when unknown (censored before H)."""
    Y = []
    for H in HORIZONS:
        b = H // SEG
        y = torch.where((ev_bin >= 0) & (ev_bin < b), 1, torch.where(n_obs >= b, 0, -1)) if ev_bin.dim() else None
        Y.append(y)
    return torch.stack(Y, 1)


# ---------------------------------------------------------------- training / eval
def batches(S, bs, shuffle):
    idx = np.random.permutation(len(S)) if shuffle else np.arange(len(S))
    for i in range(0, len(S), bs): yield idx[i:i + bs]


def make_inputs(trajs, S, idx, X, ctrl=None):
    """Gather the W most recent blocks ending at the cutoff (causal: never beyond block c*R)."""
    xs, ms = [], []
    for j in idx:
        ti, c = S[j]["t"], S[j]["c"]
        end = c * R + 1                                  # blocks [0, end) are <= cutoff t = 64c
        assert end <= X[ti].shape[0] and end - 1 <= c * R, "future leakage"
        st = max(0, end - W)
        x = X[ti][st:end]; pad = W - x.shape[0]
        xs.append(F.pad(x, (0, 0, pad, 0))); ms.append(torch.cat([torch.zeros(pad), torch.ones(x.shape[0])]))
    xs = torch.stack(xs).to(dev); ms = torch.stack(ms).to(dev)
    if ctrl == "zero": xs = torch.zeros_like(xs)
    if ctrl == "shuffle": xs = xs[torch.randperm(xs.shape[0], device=dev)];
    return xs, ms


def tens(S, idx, key, dt=torch.long):
    return torch.tensor(np.array([S[j][key] for j in idx]), dtype=dt, device=dev)


def train_model(model, trajs, Str, X, jepaT, epochs=8, bs=64, aux=True, lr=3e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    for ep in range(epochs):
        for idx in batches(Str, bs, True):
            xs, ms = make_inputs(trajs, Str, idx, X)
            o = model(xs, ms)
            evb, nob = tens(Str, idx, "ev_bin"), tens(Str, idx, "n_obs")
            safe = (evb != 0) | True
            if model.hazard:
                loss = hazard_nll(o["haz_logit"], evb, nob)
            else:
                Y = horizon_labels(evb, nob).float(); mk = (Y >= 0).float()
                loss = (F.binary_cross_entropy_with_logits(o["cls_logit"], Y.clamp(min=0), reduction="none") * mk).sum() / mk.sum().clamp(min=1)
            beh, top = tens(Str, idx, "beh"), tens(Str, idx, "top")
            loss = loss + F.cross_entropy(o["beh"].reshape(-1, len(BEH)), beh.reshape(-1), ignore_index=-1)
            loss = loss + F.cross_entropy(o["top"].reshape(-1, NTOP), top.reshape(-1), ignore_index=-1)
            loss = loss + (1 - F.cosine_similarity(o["summ"], tens(Str, idx, "summ", torch.float), -1)).mean()
            if aux and "jepa" in o:   # future-embedding prediction of target's own pooled future-window activations
                tgt = torch.stack([jepaT[(Str[j]["t"], Str[j]["c"])] for j in idx]).to(dev)   # [B, NB, dj]
                tm = torch.stack([jepaM[(Str[j]["t"], Str[j]["c"])] for j in idx]).to(dev)
                l = (1 - F.cosine_similarity(o["jepa"], tgt, -1)) * tm
                loss = loss + l.sum() / tm.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model


@torch.no_grad()
def predict(model, trajs, S, X, ctrl=None):
    model.eval(); P = dict(risk=[], beh=[], top=[], summ=[])
    for idx in batches(S, 256, False):
        xs, ms = make_inputs(trajs, S, idx, X, ctrl)
        o = model(xs, ms)
        P["risk"].append((cum_risk(o["haz_logit"]) if model.hazard else torch.sigmoid(o["cls_logit"])).cpu())
        P["beh"].append(o["beh"].argmax(-1).cpu()); P["top"].append(o["top"].argmax(-1).cpu())
        P["summ"].append(F.normalize(o["summ"], dim=-1).cpu())
    P = {k: torch.cat(v).numpy() for k, v in P.items()}
    if np.isnan(P["risk"]).any(): print("WARNING NaN risk", ctrl, np.isnan(P["risk"]).mean()); P["risk"] = np.nan_to_num(P["risk"], nan=0.5)
    return P


jepaM = {}


def boot_ci(fn, fam, n=300, seed=0):
    """Bootstrap CI resampling prompt FAMILIES (cluster bootstrap)."""
    rng = np.random.default_rng(seed); uf = np.unique(fam); vals = []
    for _ in range(n):
        pick = rng.choice(uf, len(uf)); idx = np.concatenate([np.where(fam == f)[0] for f in pick])
        try: vals.append(fn(idx))
        except Exception: pass
    vals = [v for v in vals if v == v]
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def safety_metrics(risk, Y, fam, ci=True):
    out = {}
    for h, H in enumerate(HORIZONS):
        k = Y[:, h] >= 0; y = Y[k, h]; p = np.clip(risk[k, h], 1e-6, 1 - 1e-6); f = fam[k]
        r = dict(n=int(k.sum()), pos=int(y.sum()))
        if 0 < y.sum() < len(y):
            r["auroc"] = float(roc_auc_score(y, p)); r["auprc"] = float(average_precision_score(y, p))
            r["brier"] = float(np.mean((p - y) ** 2)); r["logloss"] = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
            if ci: r["auroc_ci"] = boot_ci(lambda i: roc_auc_score(y[i], p[i]) if 0 < y[i].sum() < len(i) else np.nan, f)
        out[H] = r
    return out


def early_warning(risk, Ste, trajs, far=0.05, hcol=3):
    """Trajectory-level early warning: alarm when F(H=512) exceeds a threshold set so 5% of
    no-event test trajectories ever alarm. Recall = fraction of event trajectories alarmed at least
    L tokens before onset, for L in {128,256,512,1024}."""
    by = {}
    for j, s in enumerate(Ste): by.setdefault(s["t"], []).append((s["c"], risk[j, hcol]))
    neg = [max(v for _, v in by[t]) for t in by if trajs[t]["e"] is None]
    if len(neg) < 5: return {}
    thr = np.quantile(neg, 1 - far); res = {}
    pos = [t for t in by if trajs[t]["e"] is not None and trajs[t]["e"] > 0]
    for L in [128, 256, 512, 1024]:
        hit = elig = 0
        for t in pos:
            e = trajs[t]["e"] * SEG
            if e < L: continue
            elig += 1
            hit += any(v > thr and e - c * SEG >= L for c, v in by[t])
        res[L] = dict(recall=hit / elig if elig else np.nan, n=elig)
    return res


def main():
    os.makedirs(RES, exist_ok=True)
    seeds = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
    G, A, acts, ids = load()
    trajs, topics_all = build(G, A, acts, ids)
    tr_ids = [k for k, t in enumerate(trajs) if t["split"] == "train"]
    # --- split audit: no family in both splits
    ftr = {trajs[k]["family"] for k in tr_ids}; fte = {t["family"] for t in trajs if t["split"] == "test"}
    assert not (ftr & fte), "family leakage"
    # --- topic clusters (fit on TRAIN topics only)
    uniq = sorted(set(topics_all)); E = minilm(uniq); emb = dict(zip(uniq, E))
    trtop = sorted({s["topic"] or "none" for k in tr_ids for s in trajs[k]["segs"]})
    km = KMeans(NTOP, random_state=0, n_init=4).fit(np.stack([emb[t] for t in trtop]))
    topic_lab = {t["id"]: km.predict(np.stack([emb[s["topic"] or "none"] for s in t["segs"]])) for t in trajs}
    # --- samples at fixed cutoffs; remaining-response summary target = descs of future segments
    S, summ_txt = [], []
    for ti, t in enumerate(trajs):
        for c in range(0, min(t["K"], NB)):
            S.append(dict(t=ti, c=c)); summ_txt.append(" ".join(s["desc"] for s in t["segs"][c:]))
    SE = minilm(summ_txt); summ_emb = {(trajs[s["t"]]["id"], s["c"]): SE[j] for j, s in enumerate(S)}
    for s in S: s.update(targets_for(trajs[s["t"]], s["c"], topic_lab, summ_emb))
    # --- inputs: per-layer standardization with TRAIN stats; JEPA targets = PCA-128 of target's own future window activations
    X = [torch.tensor(t["x"]).flatten(1) for t in trajs]
    Xtr = torch.cat([X[k] for k in tr_ids]); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-3
    X = [((x - mu) / sd).to(dev) for x in X]
    U, Sv, V = torch.pca_lowrank(torch.cat([X[k] for k in tr_ids])[::4], q=128, center=False)
    for s in S:
        x = X[s["t"]]; tg = torch.zeros(NB, 128); mk = torch.zeros(NB)
        for k in range(NB):
            a, b = 1 + (s["c"] + k) * R, 1 + (s["c"] + k + 1) * R      # blocks of future segment c+k (target only)
            if b <= x.shape[0]: tg[k] = (x[a:b].mean(0) @ V).cpu(); mk[k] = 1
        jepaT_[(s["t"], s["c"])] = tg; jepaM[(s["t"], s["c"])] = mk
    din = X[0].shape[1]
    # main safety analysis: still-safe prefixes only (event not before cutoff)
    def still_safe(s): t = trajs[s["t"]]; return t["e"] is None or t["e"] >= s["c"]
    Str = [s for s in S if trajs[s["t"]]["split"] == "train" and still_safe(s)]
    Ste = [s for s in S if trajs[s["t"]]["split"] == "test" and still_safe(s)]
    Y = horizon_labels(torch.tensor([s["ev_bin"] for s in Ste]), torch.tensor([s["n_obs"] for s in Ste])).numpy()
    # secondary eval: generations that ended with EOS and no event are treated as event-free
    Y_eos = Y.copy()
    for j, s in enumerate(Ste):
        if trajs[s["t"]]["e"] is None and trajs[s["t"]]["eos"]: Y_eos[j][Y_eos[j] < 0] = 0
    fam = np.array([trajs[s["t"]]["family"] for s in Ste])
    stats = dict(n_traj=len(trajs), n_train_traj=len(tr_ids), n_test_traj=len(trajs) - len(tr_ids),
                 n_events=sum(t["e"] is not None for t in trajs), onset_tokens=[t["e"] * SEG for t in trajs if t["e"] is not None],
                 n_train_samples=len(Str), n_test_samples=len(Ste), event=EVENT,
                 test_pos_per_H={H: int((Y[:, h] == 1).sum()) for h, H in enumerate(HORIZONS)},
                 test_known_per_H={H: int((Y[:, h] >= 0).sum()) for h, H in enumerate(HORIZONS)},
                 mean_len=float(np.mean([len(t["gen_ids"]) for t in trajs])), frac_eos=float(np.mean([t["eos"] for t in trajs])))
    print(json.dumps(stats)[:600], flush=True)
    results = dict(stats=stats, safety={}, safety_eos={}, content={}, ew={}, cost={})

    def content_metrics(P, name, Sx):
        beh = np.stack([s["beh"] for s in Sx]); top = np.stack([s["top"] for s in Sx])
        acc_b = [float((P["beh"][:, k] == beh[:, k])[beh[:, k] >= 0].mean()) for k in range(KC)]
        acc_t = [float((P["top"][:, k] == top[:, k])[top[:, k] >= 0].mean()) for k in range(KC)]
        T = np.stack([s["summ"] for s in Sx]); sim = P["summ"] @ T.T
        rank = (sim > np.diag(sim)[:, None]).sum(1)
        results["content"][name] = dict(beh_acc=acc_b, top_acc=acc_t, summ_cos=float(np.diag(sim).mean()),
                                        summ_mrr=float(np.mean(1 / (1 + rank))), summ_top10=float(np.mean(rank < 10)))

    def record(name, risk):
        results["safety"][name] = safety_metrics(risk, Y, fam)
        results["safety_eos"][name] = safety_metrics(risk, Y_eos, fam, ci=False)
        results["ew"][name] = early_warning(risk, Ste, trajs)

    # ---------------- baselines
    Ytr = horizon_labels(torch.tensor([s["ev_bin"] for s in Str]), torch.tensor([s["n_obs"] for s in Str])).numpy()
    # prevalence: Kaplan-Meier style hazard per bin from train (censoring-aware)
    ev = np.array([s["ev_bin"] for s in Str]); no = np.array([s["n_obs"] for s in Str])
    hz = np.array([((ev == k).sum() + 0.5) / ((no > k).sum() + 1) for k in range(NB)])
    km_r = np.array([1 - np.prod(1 - hz[:H // SEG]) for H in HORIZONS])
    record("prevalence", np.tile(km_r, (len(Ste), 1)))
    def feat_pos(Sx): return np.array([[s["c"], np.log1p(s["c"])] for s in Sx])
    def feat_last(Sx): return torch.stack([X[s["t"]][s["c"] * R] for s in Sx]).cpu().numpy() @ V.cpu().numpy()
    def feat_text(Sx, vec=None):
        txt = [trajs[s["t"]]["prompt"] + " ||| " + tok_dec(trajs[s["t"]]["gen_ids"][:s["c"] * SEG]) for s in Sx]
        return txt
    def per_h(name, ftr_, fte_):
        risk = np.zeros((len(Ste), len(HORIZONS)))
        for h in range(len(HORIZONS)):
            k = Ytr[:, h] >= 0
            if len(np.unique(Ytr[k, h])) < 2: risk[:, h] = Ytr[k, h].mean() if k.any() else 0; continue
            clf = LogisticRegression(C=0.05, max_iter=2000).fit(ftr_[k], Ytr[k, h]); risk[:, h] = clf.predict_proba(fte_)[:, 1]
        risk = np.maximum.accumulate(risk, 1)   # enforce monotonic in H for fairness
        record(name, risk)
    per_h("position", feat_pos(Str), feat_pos(Ste))
    per_h("linear_probe", feat_last(Str), feat_last(Ste))
    tv = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
    Ttr = tv.fit_transform(feat_text(Str)); Tte = tv.transform(feat_text(Ste))
    per_h("text_only_tfidf", Ttr, Tte)
    # content baselines: per-distance majority from train
    Sc_tr = [s for s in S if trajs[s["t"]]["split"] == "train"]; Sc_te = [s for s in S if trajs[s["t"]]["split"] == "test"]
    bt = np.stack([s["beh"] for s in Sc_tr]); tt = np.stack([s["top"] for s in Sc_tr])
    maj_b = [np.bincount(bt[bt[:, k] >= 0, k], minlength=len(BEH)).argmax() for k in range(KC)]
    maj_t = [np.bincount(tt[tt[:, k] >= 0, k], minlength=NTOP).argmax() for k in range(KC)]
    msumm = np.stack([s["summ"] for s in Sc_tr]).mean(0); msumm /= np.linalg.norm(msumm)
    content_metrics(dict(beh=np.tile(maj_b, (len(Sc_te), 1)), top=np.tile(maj_t, (len(Sc_te), 1)),
                         summ=np.tile(msumm, (len(Sc_te), 1))), "majority", Sc_te)
    # current-segment copy baseline: predict every future segment = behavior/topic of the latest observed segment
    def copy_last(Sx):
        b = np.array([BEH.index(trajs[s["t"]]["segs"][s["c"] - 1]["behavior"]) if s["c"] > 0 else maj_b[0] for s in Sx])
        tp = np.array([topic_lab[trajs[s["t"]]["id"]][s["c"] - 1] if s["c"] > 0 else maj_t[0] for s in Sx])
        return dict(beh=np.repeat(b[:, None], KC, 1), top=np.repeat(tp[:, None], KC, 1), summ=np.tile(msumm, (len(Sx), 1)))
    content_metrics(copy_last(Sc_te), "copy_last_segment(oracle-annot)", Sc_te)
    json.dump(results, open(f"{RES}/results_{EVENT}.json", "w"), indent=1, default=float)

    # ---------------- neural observers (matched inputs, heads, epochs; params reported)
    global jepaT
    jepaT = jepaT_
    dj = 128
    configs = {
        "direct_AO": lambda: (DirectAO(din, dj), False),
        "A_horizon_query+mem+jepa": lambda: (HorizonQuery(din, dj), True),
        "B_latent_rollout+mem+jepa": lambda: (Rollout(din, dj), True),
        "D_hierarchical+mem+jepa": lambda: (Hier(din, dj), True),
        "A_no_memory": lambda: (HorizonQuery(din, dj, use_memory=False), True),
        "A_no_jepa": lambda: (HorizonQuery(din, dj), False),
        "A_no_hazard": lambda: (HorizonQuery(din, dj, hazard=False), True),
        "B_no_memory": lambda: (Rollout(din, dj, use_memory=False), True),
    }
    only = os.environ.get("ONLY")
    allrisk = {}
    for name, mk in configs.items():
        if only and name not in only.split(","): continue
        per_seed = []
        for sd_ in seeds:
            set_seed(sd_); model, aux = mk(); model = model.to(dev)
            t0 = time.time()
            # train on safety samples (still-safe) + content for all train cutoffs
            train_model(model, trajs, Sc_tr if False else Str, X, jepaT, aux=aux)
            tt_ = time.time() - t0
            risk = predict(model, trajs, Ste, X)["risk"]; per_seed.append(risk)
            Pc = predict(model, trajs, Sc_te, X)
            content_metrics(Pc, f"{name}/seed{sd_}", Sc_te)
            if sd_ == seeds[0]:
                for ctrl in ["zero", "shuffle"]:
                    record(f"{name}/ctrl_{ctrl}", predict(model, trajs, Ste, X, ctrl)["risk"])
                    content_metrics(predict(model, trajs, Sc_te, X, ctrl), f"{name}/ctrl_{ctrl}", Sc_te)
                xs, ms = make_inputs(trajs, Ste, np.arange(1), X); torch.cuda.synchronize(); t1 = time.time()
                with torch.no_grad():
                    for _ in range(20): model(xs, ms)
                torch.cuda.synchronize()
                results["cost"][name] = dict(params=sum(p.numel() for p in model.parameters()), train_s=tt_,
                                             latency_ms=(time.time() - t1) / 20 * 1000)
            record(f"{name}/seed{sd_}", risk)
        allrisk[name] = per_seed
        print(name, {H: results["safety"][f"{name}/seed{seeds[0]}"][H].get("auroc") for H in HORIZONS}, flush=True)
        json.dump(results, open(f"{RES}/results_{EVENT}.json", "w"), indent=1, default=float)
    np.save(f"{RES}/risk_{EVENT}.npy", {k: np.stack(v) for k, v in allrisk.items()}, allow_pickle=True)
    np.save(f"{RES}/test_meta_{EVENT}.npy", dict(Y=Y, Y_eos=Y_eos, fam=fam, c=np.array([s["c"] for s in Ste])), allow_pickle=True)
    print("done")


jepaT_ = {}
_tok = None
def tok_dec(ids):
    global _tok
    if _tok is None: _tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    return _tok.decode(ids)


if __name__ == "__main__":
    main()
