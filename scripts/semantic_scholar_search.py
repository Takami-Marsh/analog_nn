#!/usr/bin/env python3
"""Semantic Scholar search helper (requires SEMANTIC_SCHOLAR_API_KEY)."""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import requests


def search(query: str, max_results: int) -> List[Dict]:
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if not api_key:
        raise RuntimeError("SEMANTIC_SCHOLAR_API_KEY not set")
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,authors,year,venue,externalIds,url,abstract",
    }
    headers = {"x-api-key": api_key}
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    results = []
    for paper in data.get("data", []):
        authors = [a["name"] for a in paper.get("authors", [])]
        doi = (paper.get("externalIds") or {}).get("DOI", "")
        results.append(
            {
                "title": paper.get("title", ""),
                "authors": authors,
                "year": paper.get("year"),
                "doi": doi,
                "url": paper.get("url"),
                "venue": paper.get("venue"),
                "summary": (paper.get("abstract") or "")[:500],
                "source": "semantic_scholar",
                "query": query,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Query Semantic Scholar and dump metadata as JSON")
    parser.add_argument("--query", required=True)
    parser.add_argument("--max_results", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("results/semantic_scholar_results.json"))
    args = parser.parse_args()
    records = search(args.query, args.max_results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
