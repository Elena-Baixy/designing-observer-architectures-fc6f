import arxiv, sys
client = arxiv.Client(page_size=15, delay_seconds=3, num_retries=3)
seen=set()
for q in sys.argv[1:]:
    print("==", q)
    try:
        for r in client.results(arxiv.Search(query=q, max_results=12, sort_by=arxiv.SortCriterion.Relevance)):
            i=r.get_short_id()
            if i in seen: continue
            seen.add(i); print(f"  {i} | {r.published.date()} | {r.title[:110]}")
    except Exception as e: print("ERR", e)
