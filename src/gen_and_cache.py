"""Steps 1-2: build prompts from existing benchmark datasets (unmodified, no jailbreak framing),
assign prompt families and the train/test split BEFORE generation, generate long continuations
from frozen Qwen3-8B, then cache multi-layer block-pooled activations with one causal forward pass.

Observer inputs = means of residual activations over 16-token blocks of the continuation. Qwen3 is
causal, so the activation at token t depends only on tokens <= t; at cutoff t the observer only reads
blocks that end at or before t (audited in train_observers.py).
"""
import json, random, os, time
import numpy as np, torch, datasets
from transformers import AutoTokenizer, AutoModelForCausalLM

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
OUT = os.environ.get("OUT", "artifacts")
MAXNEW = int(os.environ.get("MAXNEW", 2048))
BS = int(os.environ.get("BS", 64))
N_PER = int(os.environ.get("N_PER", 48))
LAYERS = [9, 18, 27, 35]
BLOCK = 16
MODEL = "Qwen/Qwen3-8B"


def build_prompts():
    rng = random.Random(SEED)
    P = []
    lw = datasets.load_from_disk("datasets/longwriter_6k")["train"]
    idx = rng.sample(range(len(lw)), 4 * N_PER)
    for i in idx:
        q = lw[i]["messages"][0]["content"]
        if len(q) < 1500:
            P.append(dict(src="longwriter", family=f"longwriter_{i % 16}", prompt=q))
    P = P[: N_PER + 16]
    tc = datasets.load_from_disk("datasets/toxic_chat")["train"]
    tc = [r for r in tc if r["toxicity"] == 1 and len(r["user_input"]) < 1500]
    for r in rng.sample(tc, N_PER):
        P.append(dict(src="toxic_chat", family=f"toxic_chat_jb{r['jailbreaking']}_{hash(r['conv_id']) % 4}", prompt=r["user_input"]))
    bt = datasets.load_from_disk("datasets/beavertails")["30k_train"]
    bt = [r for r in bt if not r["is_safe"]]
    for r in rng.sample(bt, N_PER):
        cat = sorted([k for k, v in r["category"].items() if v])[0]
        P.append(dict(src="beavertails", family=f"beavertails_{cat}", prompt=r["prompt"]))
    ag = datasets.load_from_disk("datasets/aegis2")["train"]
    ag = [r for r in ag if r["prompt_label"] == "unsafe" and r["prompt"] != "REDACTED" and len(r["prompt"]) < 1500]
    for r in rng.sample(ag, N_PER):
        cat = (r["violated_categories"] or "none").split(",")[0].strip()
        P.append(dict(src="aegis2", family=f"aegis2_{cat}", prompt=r["prompt"]))
    xs = datasets.load_from_disk("datasets/xstest")["train"]
    for r in rng.sample(list(xs), N_PER):
        P.append(dict(src="xstest", family=f"xstest_{r['type']}", prompt=r["prompt"]))
    # split by family before generation
    fams = sorted(set(p["family"] for p in P))
    rng.shuffle(fams)
    test_f = set(fams[: int(0.3 * len(fams))])
    for i, p in enumerate(P):
        p["id"] = i
        p["split"] = "test" if p["family"] in test_f else "train"
    return P


def main():
    os.makedirs(OUT, exist_ok=True)
    P = build_prompts()
    print(len(P), "prompts;", sum(p["split"] == "test" for p in P), "test")
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    for p in P:
        p["chat"] = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
    gen_path = f"{OUT}/generations.jsonl"
    done = set()
    if os.path.exists(gen_path):
        done = {json.loads(l)["id"] for l in open(gen_path)}
    todo = [p for p in P if p["id"] not in done]
    for b in range(0, len(todo), BS):
        batch = todo[b:b + BS]
        t0 = time.time()
        enc = tok([p["chat"] for p in batch], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAXNEW, do_sample=True, temperature=0.7, top_p=0.9,
                                 pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:].cpu()
        with open(gen_path, "a") as f:
            for p, g in zip(batch, gen):
                g = g.tolist()
                eos = [i for i, t in enumerate(g) if t in (tok.eos_token_id, tok.pad_token_id, 151645)]
                L = eos[0] if eos else len(g)
                ids = g[:L]
                f.write(json.dumps(dict(id=p["id"], src=p["src"], family=p["family"], split=p["split"],
                                        prompt=p["prompt"], prompt_ids=tok(p["chat"])["input_ids"],
                                        gen_ids=ids, ended_eos=bool(eos), text=tok.decode(ids))) + "\n")
        print(f"batch {b // BS} done in {time.time() - t0:.0f}s", flush=True)
    # ---- activation cache (causal forward pass, hooks on chosen layers) ----
    G = [json.loads(l) for l in open(gen_path)]
    acts = {}
    store = {}
    hooks = [model.model.layers[l].register_forward_hook(
        lambda m, i, o, l=l: store.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[0])) for l in LAYERS]
    for g in G:
        ids = g["prompt_ids"] + g["gen_ids"]
        with torch.no_grad():
            model(torch.tensor([ids], device="cuda"))
        P0 = len(g["prompt_ids"])
        nb = len(g["gen_ids"]) // BLOCK
        H = torch.stack([store[l] for l in LAYERS], 0)  # [4, T, d]
        pr = H[:, P0 - 8:P0].mean(1, keepdim=True)  # prompt-end summary = "block 0"
        if nb > 0:
            blk = H[:, P0:P0 + nb * BLOCK].reshape(len(LAYERS), nb, BLOCK, -1).mean(2)
            blk = torch.cat([pr, blk], 1)
        else:
            blk = pr
        acts[g["id"]] = blk.transpose(0, 1).float().cpu().half().numpy()  # [nb+1, 4, d]
    for h in hooks:
        h.remove()
    np.savez(f"{OUT}/acts.npz", **{str(k): v for k, v in acts.items()})
    print("cached acts for", len(acts))


if __name__ == "__main__":
    main()
