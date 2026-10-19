import arxiv, sys, json
client = arxiv.Client(page_size=50, delay_seconds=3, num_retries=3)
ids=sys.argv[1:]
res=list(client.results(arxiv.Search(id_list=ids)))
meta=[]
for r in res:
    meta.append(dict(id=r.get_short_id().split('v')[0], title=r.title, authors=[a.name for a in r.authors][:6], year=r.published.year, abstract=r.summary.replace('\n',' ')))
    print(f"### {r.get_short_id()} {r.title} ({r.published.year})\n{r.summary.replace(chr(10),' ')[:900]}\n")
json.dump(meta, open('papers/arxiv_meta.json','w'), indent=1)
