#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -U pip wheel >/dev/null
pip install -r requirements.txt >/dev/null

python scripts/fetch_papers.py --config docs/search_queries.yml --out_raw results/papers_raw.json --out_dedup results/papers_deduped.json
python scripts/build_bib.py results/papers_deduped.json references.bib docs/references.md
python src/run_toy_sim.py --config config.yml --results results/toy_mse_vs_snr.csv --figure figures/toy_mse_vs_snr.png
python src/run_digits_demo.py --config config.yml --results results/acc_vs_noise.csv --figure figures/acc_vs_noise.png
python scripts/build_paper_html.py --output results/capstone_report.html
python scripts/html_to_pdf.py results/capstone_report.html results/capstone_report.pdf

echo "Reproduction complete."
