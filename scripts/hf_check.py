import sys, requests
for d in sys.argv[1:]:
    r = requests.get(f"https://huggingface.co/api/datasets/{d}", timeout=30)
    if r.status_code!=200: print(d, "->", r.status_code); continue
    j=r.json()
    sz = None
    try:
        s=requests.get(f"https://datasets-server.huggingface.co/size?dataset={d}",timeout=30).json()
        sz = s.get('size',{}).get('dataset',{})
        sz = (sz.get('num_rows'), round(sz.get('num_bytes_original_files',0)/1e6,1))
    except Exception as e: pass
    print(d, "| gated:", j.get('gated'), "| downloads:", j.get('downloads'), "| rows,MB:", sz)
