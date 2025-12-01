#!/usr/bin/env python3
"""
Render an HTML file to PDF. Prefers system Google Chrome headless; falls back to pyppeteer.
"""
import argparse
import asyncio
import subprocess
from pathlib import Path

from pyppeteer import launch
from pyppeteer.chromium_downloader import chromium_executable


def chrome_path() -> Path:
    candidates = [
        Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return Path()


def render_with_chrome(html_path: Path, pdf_path: Path) -> bool:
    chrome = chrome_path()
    if not chrome:
        return False
    cmd = [
        str(chrome),
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        f"--print-to-pdf={pdf_path}",
        "--print-to-pdf-no-header",
        f"file://{html_path.resolve()}",
    ]
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)
    return True


async def render_with_pyppeteer(html_path: Path, pdf_path: Path) -> None:
    browser = await launch(
        headless=True,
        executablePath=chromium_executable(),
        args=["--no-sandbox", "--disable-dev-shm-usage"],
    )
    page = await browser.newPage()
    await page.goto(f"file://{html_path.resolve()}", waitUntil="networkidle0")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    await page.pdf(
        {
            "path": str(pdf_path),
            "format": "A4",
            "printBackground": True,
            "margin": {"top": "10mm", "bottom": "15mm", "left": "12mm", "right": "12mm"},
        }
    )
    await browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render HTML to PDF (Chrome or pyppeteer)")
    parser.add_argument("html", type=Path, help="Input HTML file")
    parser.add_argument("pdf", type=Path, help="Output PDF file")
    args = parser.parse_args()
    try:
        used_chrome = render_with_chrome(args.html, args.pdf)
        if not used_chrome:
            asyncio.get_event_loop().run_until_complete(render_with_pyppeteer(args.html, args.pdf))
        print(f"Wrote {args.pdf}")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Failed to render PDF: {exc}")


if __name__ == "__main__":
    main()
