#!/usr/bin/env python3
"""Standalone arXiv search helper."""
import argparse
import json
from pathlib import Path
from typing import List, Dict

import arxiv


def search(query: str, max_results: int) -> List[Dict]:
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    results = []
    for result in client.results(search):
        results.append(
            {
                "title": result.title,
                "authors": [str(a) for a in result.authors],
                "year": result.published.year,
                "doi": result.doi or "",
                "url": result.entry_id,
                "summary": result.summary[:500],
                "source": "arxiv",
                "query": query,
                "arxiv_id": result.get_short_id(),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Search arXiv and dump metadata as JSON")
    parser.add_argument("--query", required=True, help="Search string")
    parser.add_argument("--max_results", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("results/arxiv_results.json"))
    args = parser.parse_args()
    records = search(args.query, args.max_results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
