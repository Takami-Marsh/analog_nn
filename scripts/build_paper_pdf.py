#!/usr/bin/env python3
"""
Generate a PDF "paper" by stitching key docs and figures.
"""
import argparse
from pathlib import Path
from typing import Iterable, List

from fpdf import FPDF
from fpdf.errors import FPDFException


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

FIGURES = [
    ("Toy MAC MSE vs SNR", "figures/toy_mse_vs_snr.png"),
    ("Digits accuracy vs phase noise", "figures/acc_vs_noise.png"),
]


class PaperPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "Analog / Wave-Based Neural Network Capstone", new_x="LMARGIN", new_y="NEXT", align="C")
        self.ln(2)

def footer(self):
    self.set_y(-15)
    self.set_font("Helvetica", "I", 8)
    self.cell(0, 10, f"Page {self.page_no()}", align="C")


def soft_break(text: str, max_len: int = 60) -> str:
    tokens = text.split(" ")
    broken: List[str] = []
    for tok in tokens:
        if len(tok) <= max_len:
            broken.append(tok)
        else:
            for i in range(0, len(tok), max_len):
                broken.append(tok[i : i + max_len])
    return " ".join(broken)


def clean_text(text: str) -> str:
    replacements = {
        "\u2014": "-",
        "\u2013": "-",
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
    for src, tgt in replacements.items():
        text = text.replace(src, tgt)
    # Drop any remaining non-ASCII
    return text.encode("ascii", "ignore").decode("ascii")


def safe_multi(pdf: PaperPDF, h: float, text: str) -> None:
    try:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, h, text)
    except FPDFException:
        truncated = text[:80] + "..." if len(text) > 80 else text
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, h, truncated)


def add_markdown(pdf: PaperPDF, text: str) -> None:
    for raw_line in text.splitlines():
        line = clean_text(raw_line.rstrip())
        if not line:
            pdf.ln(4)
            continue
        if line.startswith("### "):
            pdf.set_font("Helvetica", "B", 11)
            safe_multi(pdf, 6, soft_break(line[4:]))
        elif line.startswith("## "):
            pdf.set_font("Helvetica", "B", 12)
            safe_multi(pdf, 7, soft_break(line[3:]))
            pdf.ln(2)
        elif line.startswith("# "):
            pdf.set_font("Helvetica", "B", 13)
            safe_multi(pdf, 8, soft_break(line[2:]))
            pdf.ln(3)
        elif line.startswith("|"):
            # Table row (e.g., references); simplify to avoid long unbreakable URLs
            if line.startswith("| ---"):
                continue
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) >= 3:
                title_cell = parts[0]
                year_cell = parts[1]
                source_cell = parts[2]
                if "]" in title_cell:
                    title_cell = title_cell.split("]")[0].lstrip("[").strip()
                pdf.set_font("Helvetica", "", 10)
                summary = f"- {title_cell} ({year_cell}) [{source_cell}]"
                safe_multi(pdf, 5, soft_break(summary))
            continue
        elif line.startswith("- "):
            pdf.set_font("Helvetica", "", 10)
            safe_multi(pdf, 5, soft_break("- " + line[2:]))
        else:
            pdf.set_font("Helvetica", "", 10)
            safe_multi(pdf, 5, soft_break(line))


def embed_figures(pdf: PaperPDF, figs: Iterable) -> None:
    for caption, path in figs:
        img_path = Path(path)
        if not img_path.exists():
            continue
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 12)
        pdf.multi_cell(0, 6, caption)
        pdf.ln(2)
        width = 170
        pdf.image(str(img_path), w=width)


def build_pdf(output: Path, sections: List[str], figs=FIGURES) -> None:
    pdf = PaperPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    for section in sections:
        path = Path(section)
        if not path.exists():
            continue
        pdf.add_page()
        add_markdown(pdf, path.read_text(encoding="utf-8"))
    embed_figures(pdf, figs)
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build capstone PDF")
    parser.add_argument(
        "--output", type=Path, default=Path("results/capstone_report.pdf"), help="Output PDF path"
    )
    parser.add_argument(
        "--sections",
        nargs="*",
        default=SECTION_FILES,
        help="List of markdown files to include in order",
    )
    args = parser.parse_args()
    build_pdf(args.output, args.sections)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
