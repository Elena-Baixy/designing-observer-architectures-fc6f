import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
def q(a): a=np.array(a); return dict(n=len(a), mean=int(a.mean()), p10=int(np.percentile(a,10)), p50=int(np.median(a)), p90=int(np.percentile(a,90)), max=int(a.max()))
g = load_from_disk("datasets/qwen3guardtest")
for split in ["thinking_loc","response_loc"]:
    d=g[split]; onset=[]; total=[]
    for r in d:
        ids=r["input_ids"]
        # assistant start = last occurrence of <|im_start|>assistant (151644, 77091)
        a=max(i for i in range(len(ids)-1) if ids[i]==151644 and ids[i+1]==77091)
        onset.append(r["unsafe_start_index"]-a); total.append(len(ids)-a)
    print(split, "onset tokens into response:", q(onset)); print(split, "response length:", q(total))
    print(split, "frac onset>=256/512/1024/2048:", [round(np.mean(np.array(onset)>=t),3) for t in (256,512,1024,2048)])
lw = load_from_disk("datasets/longwriter_6k")["train"]
L=[len(tok(m[-1]["content"]).input_ids) for m in lw.select(range(500))["messages"]]
print("LongWriter-6k (500 sample) output tokens:", q(L))
cb = load_from_disk("datasets/circuit_breakers")["train"]
L=[len(tok(x).input_ids) for x in cb.select(range(500))["rejected"]]
print("circuit_breakers harmful completion tokens:", q(L))
bt = load_from_disk("datasets/beavertails")["30k_test"]
L=[len(tok(x).input_ids) for x in bt.select(range(1000))["response"]]
print("BeaverTails response tokens:", q(L))
