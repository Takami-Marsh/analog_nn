#!/usr/bin/env python3
"""
Build an HTML report by stitching Markdown docs and figures, ready for MathJax rendering.
"""
import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import markdown

SECTION_FILES = [
    "docs/08_capstone_report.md",
    "docs/01_problem_statement.md",
    "docs/02_literature_review.md",
    "docs/03_architecture_options.md",
    "docs/04_math_model.md",
    "docs/05_noise_and_variation.md",
    "docs/06_simulation_results.md",
    "docs/07_circuit_mapping.md",
    "docs/claim_audit.md",
    "docs/references.md",
]

FIGURES: List[Tuple[str, str]] = [
    ("Toy MAC MSE vs SNR", "figures/toy_mse_vs_snr.png"),
    ("Digits accuracy vs phase noise", "figures/acc_vs_noise.png"),
]


def md_to_html(text: str) -> str:
    return markdown.markdown(
        text,
        extensions=[
            "extra",
            "toc",
            "tables",
            "fenced_code",
            "sane_lists",
            "nl2br",
        ],
    )


def build_html(sections: List[str], figures: Iterable[Tuple[str, str]]) -> str:
    parts = []
    for section in sections:
        path = Path(section)
        if not path.exists():
            continue
        parts.append(f"<section id='{path.stem}'>")
        parts.append(md_to_html(path.read_text(encoding='utf-8')))
        parts.append("</section>")
    # Figures
    for caption, fig_path in figures:
        p = Path(fig_path)
        if p.exists():
            parts.append("<section class='figure'>")
            parts.append(f"<h3>{caption}</h3>")
            parts.append(f"<img src='{p.as_posix()}' alt='{caption}' />")
            parts.append("</section>")
    body = "\n".join(parts)
    style = """
    body { font-family: 'Helvetica', 'Arial', sans-serif; margin: 30px; color: #111; line-height: 1.5; }
    h1,h2,h3,h4 { color: #0d1b2a; margin-top: 24px; }
    section { margin-bottom: 24px; }
    code { background: #f4f4f4; padding: 2px 4px; border-radius: 3px; }
    pre { background: #f8f8f8; padding: 10px; border-radius: 6px; overflow-x: auto; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 6px 8px; }
    th { background: #f0f0f0; }
    img { max-width: 100%; height: auto; }
    """
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <title>Analog / Wave-Based Neural Network Capstone</title>
  <style>{style}</style>
  <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [['$','$'], ['\\\\(','\\\\)']],
        displayMath: [['$$','$$'], ['\\\\[','\\\\]']]
      }},
      svg: {{ fontCache: 'global' }}
    }};
  </script>
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head>
<body>
{body}
</body>
</html>"""
    return html


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HTML report for capstone")
    parser.add_argument(
        "--output", type=Path, default=Path("results/capstone_report.html"), help="Output HTML path"
    )
    parser.add_argument(
        "--sections",
        nargs="*",
        default=SECTION_FILES,
        help="Ordered list of markdown files to include",
    )
    args = parser.parse_args()
    html = build_html(args.sections, FIGURES)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
