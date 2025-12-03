# Analog / Wave-Based Neural Network Capstone

Research-backed, reproducible package for phase/frequency-encoded analog neural networks. Includes literature harvesting, math models, noise/noise-aware training analysis, benchmarks, and runnable UIs.

## Quick start
```bash
bash scripts/reproduce.sh
```
This creates a Python venv, installs dependencies, fetches papers, builds the annotated bibliography, and runs simulations to regenerate figures/tables.

## Layout
- `docs/`: problem statement, literature review, architectures, math model, noise plan, simulation results, slides, claim audit, and benchmark reports (`benchmark_report.tex/pdf`, Japanese under `docs/ja/`).
- `src/`: simulations and benchmarks (`run_toy_sim.py`, `run_digits_demo.py`, `run_benchmark.py`, `run_benchmark_fashion.py`, Gradio UIs).
- `scripts/`: automation (`reproduce.sh`, `fetch_papers.py`, `build_bib.py`, `analyze_benchmark.py`).
- `figures/`, `results/`: generated plots and tables (benchmarks land here).
- `references.bib`: BibTeX citations; `docs/references.md`: annotated reference table.
- `config.yml`: shared experiment parameters.

## Dependencies
- Python 3.9+ with `venv`
- Packages pinned in `requirements.txt` (installed automatically by `reproduce.sh`)
- Optional (for PDFs): `tectonic` via Homebrew (`brew install tectonic`)

## Key commands
- Fetch/bibliography:  
  - `python scripts/fetch_papers.py --config docs/search_queries.yml`  
  - `python scripts/build_bib.py results/papers_deduped.json references.bib docs/references.md`
- Toy + digits demos:  
  - `python src/run_toy_sim.py --config config.yml`  
  - `python src/run_digits_demo.py --config config.yml`
- Benchmarks:  
  - Baseline MNIST: `python src/run_benchmark.py --config config.yml`  
  - Other datasets: `python src/run_benchmark_fashion.py --config config.yml --config-key {fashion_complex|kmnist_benchmark|emnist_letters_benchmark|cifar10_flat_benchmark}`
- UI: `python src/serve_digits_ui.py --config config.yml` (add `--share` for a public link)
- Reports (with `tectonic`):  
  - English: `cd docs && tectonic benchmark_report.tex`  
  - Japanese: `cd docs/ja && tectonic benchmark_report.tex`

## Deliverables
Core documents: `docs/01_problem_statement.md`, `docs/02_literature_review.md`, `docs/03_architecture_options.md`, `docs/04_math_model.md`, `docs/05_noise_and_variation.md`, `docs/06_simulation_results.md`, `docs/08_capstone_report.md`, `docs/09_slide_outline.md`, `docs/claim_audit.md`.
