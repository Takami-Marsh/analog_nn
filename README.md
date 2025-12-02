# Analog / Wave-Based Neural Network Capstone

Research-backed, reproducible package for phase/frequency-encoded analog neural networks. Includes literature harvesting, math models, noise analysis, and runnable simulations.

## Quick start
```bash
bash scripts/reproduce.sh
```
This creates a Python venv, installs dependencies, fetches papers, builds the annotated bibliography, and runs simulations to regenerate figures/tables.

## Layout
- `docs/`: problem statement, literature review, architectures, math model, noise plan, simulation results, report, slides, claim audit.
- `src/`: simulations (`run_toy_sim.py`, `run_digits_demo.py`).
- `scripts/`: automation (`reproduce.sh`, `fetch_papers.py`, `build_bib.py`).
- `figures/`, `results/`: generated plots and tables.
- `references.bib`: BibTeX citations; `docs/references.md`: annotated reference table.
- `config.yml`: shared experiment parameters.
- `src/serve_digits_ui.py`: interactive Gradio UI to draw digits and compare models.

## Dependencies
- Python 3.9+ with `venv`
- Packages pinned in `requirements.txt` (installed automatically by `reproduce.sh`)

## Key commands
- `python scripts/fetch_papers.py --config docs/search_queries.yml`
- `python scripts/build_bib.py results/papers_deduped.json references.bib docs/references.md`
- `python src/run_toy_sim.py --config config.yml`
- `python src/run_digits_demo.py --config config.yml`
- `python src/run_benchmark.py --config config.yml` (baseline MNIST benchmark)
- `python src/run_benchmark_fashion.py --config config.yml` (Fashion-MNIST and other dataset benchmarks; add `--config-key` e.g. `fashion_complex|kmnist_benchmark|emnist_letters_benchmark|cifar10_flat_benchmark`)
- `python src/serve_digits_ui.py --config config.yml` (launch UI; add `--share` to expose public link)

## Deliverables
Core documents: `docs/01_problem_statement.md`, `docs/02_literature_review.md`, `docs/03_architecture_options.md`, `docs/04_math_model.md`, `docs/05_noise_and_variation.md`, `docs/06_simulation_results.md`, `docs/08_capstone_report.md`, `docs/09_slide_outline.md`, `docs/claim_audit.md`.
