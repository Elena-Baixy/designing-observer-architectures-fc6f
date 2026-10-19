import arxiv, sys, json, time
client = arxiv.Client(page_size=5, delay_seconds=3, num_retries=3)
out = []
for q in sys.argv[1:]:
    try:
        res = list(client.results(arxiv.Search(query=f'ti:"{q}"', max_results=3)))
        if not res:
            res = list(client.results(arxiv.Search(query=q, max_results=3)))
    except Exception as e:
        print("ERR", q, e); continue
    for r in res[:2]:
        print(f"{q[:40]!r:45} -> {r.get_short_id()} | {r.published.year} | {r.title[:100]}")
