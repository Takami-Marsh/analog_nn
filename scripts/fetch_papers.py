#!/usr/bin/env python3
"""
Fetch paper metadata from arXiv (and optionally Semantic Scholar) using search queries.
Outputs:
- results/papers_raw.json
- results/papers_deduped.json
"""
import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Set

import arxiv
import requests
import yaml


def load_queries(path: Path) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def fetch_arxiv(query: str, max_results: int = 10) -> List[Dict]:
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    results = []
    for result in search.results():
        authors = [str(a) for a in result.authors]
        results.append(
            {
                "title": result.title,
                "authors": authors,
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


def fetch_semantic_scholar(query: str, max_results: int = 5) -> List[Dict]:
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if not api_key:
        return []
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
        doi = ""
        ext_ids = paper.get("externalIds") or {}
        if "DOI" in ext_ids:
            doi = ext_ids["DOI"]
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


def dedup(records: List[Dict]) -> List[Dict]:
    seen_doi: Set[str] = set()
    seen_title: Set[str] = set()
    deduped = []
    for rec in records:
        doi = rec.get("doi", "").lower()
        title_norm = normalize_title(rec.get("title", ""))
        if doi and doi in seen_doi:
            continue
        if doi:
            seen_doi.add(doi)
        if title_norm and title_norm in seen_title:
            continue
        if title_norm:
            seen_title.add(title_norm)
        deduped.append(rec)
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch papers for analog/wave NN capstone")
    parser.add_argument("--config", type=Path, required=True, help="Path to search_queries.yml")
    parser.add_argument("--out_raw", type=Path, default=Path("results/papers_raw.json"))
    parser.add_argument("--out_dedup", type=Path, default=Path("results/papers_deduped.json"))
    args = parser.parse_args()
    queries = load_queries(args.config)
    raw_records: List[Dict] = []
    for group, qs in queries.items():
        for q in qs:
            raw_records.extend(fetch_arxiv(q))
            raw_records.extend(fetch_semantic_scholar(q))
    deduped = dedup(raw_records)
    args.out_raw.parent.mkdir(parents=True, exist_ok=True)
    args.out_dedup.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_raw, "w", encoding="utf-8") as f:
        json.dump(raw_records, f, indent=2)
    with open(args.out_dedup, "w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2)
    print(f"raw: {len(raw_records)} papers, deduped: {len(deduped)}")


if __name__ == "__main__":
    main()
