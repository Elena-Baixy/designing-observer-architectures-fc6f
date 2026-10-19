import json, os, sys, traceback
from datasets import load_dataset, DatasetDict
OUT = "datasets"
SPECS = [  # (hf_id, config, local_name)
    ("Qwen/Qwen3GuardTest", None, "qwen3guardtest"),
    ("liyang-ict/FineHarm", None, "fineharm"),
    ("Unispac/shallow-vs-deep-safety-alignment-dataset", None, "shallow_vs_deep_safety"),
    ("abhayesian/circuit-breakers-dataset", None, "circuit_breakers"),
    ("PKU-Alignment/BeaverTails", None, "beavertails"),
    ("walledai/HarmBench", "standard", "harmbench_standard"),
    ("walledai/StrongREJECT", None, "strongreject"),
    ("walledai/AdvBench", None, "advbench"),
    ("JailbreakBench/JBB-Behaviors", "behaviors", "jbb_behaviors"),
    ("zai-org/LongWriter-6k", None, "longwriter_6k"),
    ("allenai/real-toxicity-prompts", None, "real_toxicity_prompts"),
    ("databricks/databricks-dolly-15k", None, "dolly_15k"),
    ("Paul/XSTest", None, "xstest"),
    ("LLM-LAT/harmful-dataset", None, "llm_lat_harmful"),
    ("lmsys/toxic-chat", "toxicchat0124", "toxic_chat"),
    ("AlignmentResearch/StrongREJECT", None, "strongreject"),
    ("mlabonne/harmful_behaviors", None, "harmful_behaviors_advbench"),
    ("mlabonne/harmless_alpaca", None, "harmless_alpaca"),
    ("abinzzz/ForeLen", "qwen2.5-7b-longseq", "forelen_qwen2.5_7b_longseq"),
    ("abinzzz/ForeLen", "qwen2.5-7b-reasoning", "forelen_qwen2.5_7b_reasoning"),
    ("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", None, "aegis2"),
]
only = set(sys.argv[1:])
for hf, cfg, name in SPECS:
    if only and name not in only: continue
    path = os.path.join(OUT, name)
    if os.path.exists(os.path.join(path, "dataset_dict.json")):
        print("skip", name); continue
    try:
        ds = load_dataset(hf, cfg) if cfg else load_dataset(hf)
        ds.save_to_disk(path)
        os.makedirs(os.path.join(path, "samples"), exist_ok=True)
        info = {}
        for split in ds:
            info[split] = {"rows": len(ds[split]), "columns": ds[split].column_names}
            rows = [ {k:(str(v)[:600]) for k,v in r.items() if k!='input_ids'} for r in ds[split].select(range(min(3,len(ds[split]))))]
            json.dump(rows, open(os.path.join(path, "samples", f"{split}_samples.json"), "w"), indent=1)
        json.dump(info, open(os.path.join(path, "samples", "info.json"), "w"), indent=1)
        print("OK", name, info)
    except Exception as e:
        print("FAIL", name, repr(e)[:300])
