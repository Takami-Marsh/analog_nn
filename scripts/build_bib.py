#!/usr/bin/env python3
"""
Build an annotated bibliography table and append missing BibTeX entries.
"""
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import bibtexparser


def load_existing_bib(bib_path: Path) -> Tuple[Set[str], List[Dict]]:
    if not bib_path.exists():
        return set(), []
    with open(bib_path, "r", encoding="utf-8") as f:
        db = bibtexparser.load(f)
    dois = {entry.get("doi", "").lower() for entry in db.entries if entry.get("doi")}
    titles = {normalize(entry.get("title", "")) for entry in db.entries}
    return dois, titles


def bib_entries_to_records(entries: List[Dict]) -> List[Dict]:
    records = []
    for entry in entries:
        title = entry.get("title", "")
        authors = entry.get("author", "")
        year = entry.get("year")
        venue = entry.get("journal") or entry.get("booktitle") or ""
        records.append(
            {
                "title": title,
                "authors": [a.strip() for a in authors.split(" and ")] if authors else [],
                "year": int(year) if year and str(year).isdigit() else year,
                "doi": entry.get("doi", ""),
                "url": entry.get("url", ""),
                "venue": venue,
                "summary": entry.get("note", ""),
                "source": "bib",
            }
        )
    return records


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def gen_key(title: str, year: int) -> str:
    words = [w for w in re.split(r"[^a-z0-9]+", title.lower()) if w]
    stem = "".join(words[:2]) if len(words) >= 2 else "ref"
    return f"{stem}{year}"


def record_to_bib(rec: Dict) -> Dict:
    year = rec.get("year") or "n.d."
    title = rec.get("title", "untitled")
    return {
        "ENTRYTYPE": "article",
        "ID": gen_key(title, year),
        "title": title,
        "author": " and ".join(rec.get("authors", [])) if rec.get("authors") else "Unknown",
        "year": str(year),
        "doi": rec.get("doi", ""),
        "url": rec.get("url", ""),
        "journal": rec.get("venue", rec.get("source", "unknown")),
    }


def append_missing_bib(
    bib_path: Path, records: List[Dict], existing_dois: Set[str], existing_titles: Set[str]
) -> int:
    new_entries = []
    for rec in records:
        doi = rec.get("doi", "").lower()
        title_norm = normalize(rec.get("title", ""))
        if (doi and doi in existing_dois) or (title_norm and title_norm in existing_titles):
            continue
        new_entries.append(record_to_bib(rec))
        if doi:
            existing_dois.add(doi)
        if title_norm:
            existing_titles.add(title_norm)
    if not new_entries:
        return 0
    if bib_path.exists():
        with open(bib_path, "r", encoding="utf-8") as f:
            db = bibtexparser.load(f)
    else:
        db = bibtexparser.bibdatabase.BibDatabase()
    db.entries.extend(new_entries)
    with open(bib_path, "w", encoding="utf-8") as f:
        bibtexparser.dump(db, f)
    return len(new_entries)


def dedup_records(records: List[Dict]) -> List[Dict]:
    seen: Set[str] = set()
    unique: List[Dict] = []
    for rec in records:
        title_norm = normalize(rec.get("title", ""))
        if title_norm and title_norm in seen:
            continue
        if title_norm:
            seen.add(title_norm)
        unique.append(rec)
    return unique


def build_reference_md(records: List[Dict], out_path: Path) -> None:
    lines = [
        "# References",
        "",
        "| Title | Year | Source | Note |",
        "| --- | --- | --- | --- |",
    ]
    for rec in records:
        title = rec.get("title", "Untitled")
        year = rec.get("year", "")
        source = rec.get("source", rec.get("venue", ""))
        note = (rec.get("summary", "") or "").replace("\n", " ")
        if len(note) > 140:
            note = note[:137] + "..."
        link = rec.get("url", "")
        title_cell = f"[{title}]({link})" if link else title
        lines.append(f"| {title_cell} | {year} | {source} | {note} |")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bibliography and reference table")
    parser.add_argument("input_json", type=Path, help="Deduped paper list")
    parser.add_argument("bib_path", type=Path, help="BibTeX output path")
    parser.add_argument("reference_md", type=Path, help="Markdown reference table")
    args = parser.parse_args()

    records = json.loads(args.input_json.read_text(encoding="utf-8"))
    existing_dois, existing_titles = load_existing_bib(args.bib_path)
    # Include existing BibTeX entries in the annotated reference table
    if args.bib_path.exists():
        with open(args.bib_path, "r", encoding="utf-8") as f:
            bib_db = bibtexparser.load(f)
        records = records + bib_entries_to_records(bib_db.entries)
    records = dedup_records(records)
    added = append_missing_bib(args.bib_path, records, existing_dois, existing_titles)
    build_reference_md(records, args.reference_md)
    print(f"Appended {added} BibTeX entries; wrote {len(records)} references to {args.reference_md}")


if __name__ == "__main__":
    main()
