import argparse
from pathlib import Path
from typing import Dict, List

import gradio as gr
import pandas as pd
import yaml

from run_benchmark import load_config, run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio UI to run digit benchmark")
    parser.add_argument("--config", type=Path, default=Path("config.yml"))
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    base_config = load_config(args.config)

    def run_with_override(epochs: int, noise_text: str, lr: float):
        progress = gr.Progress(track_tqdm=True)
        cfg = base_config.copy()
        cfg["digits_demo"] = cfg.get("digits_demo", {})
        cfg["digits_demo"]["epochs"] = epochs
        cfg["digits_demo"]["lr"] = lr
        if noise_text.strip():
            try:
                noise_list: List[float] = [float(tok) for tok in noise_text.replace(",", " ").split()]
                cfg["digits_demo"]["noise_std"] = noise_list
            except ValueError:
                pass
        progress(0, desc="Starting benchmark...")
        df = run_benchmark(cfg)
        summary = (
            df.groupby("model")
            .agg(
                acc_mean=("accuracy", "mean"),
                acc_min=("accuracy", "min"),
                acc_max=("accuracy", "max"),
            )
            .reset_index()
        )
        progress(1.0, desc="Done")
        return "Done", summary, df

    with gr.Blocks() as demo:
        gr.Markdown("# Benchmark: Analog / Digital Digit Classifiers\nRuns training + evaluation with configurable epochs and noise list.")
        with gr.Row():
            epochs_in = gr.Slider(5, 60, value=base_config["digits_demo"].get("epochs", 30), step=1, label="Epochs")
            lr_in = gr.Slider(0.0005, 0.05, value=base_config["digits_demo"].get("lr", 0.01), step=0.0005, label="Learning rate")
            noise_in = gr.Textbox(label="Noise std list", value=" ".join(map(str, base_config['digits_demo'].get('noise_std', []))))
            run_btn = gr.Button("Run benchmark")
        status = gr.Textbox(label="Status", interactive=False)
        summary_df = gr.Dataframe(label="Summary (mean/min/max accuracy by model)")
        full_df = gr.Dataframe(label="Full results (per noise)")
        run_btn.click(fn=run_with_override, inputs=[epochs_in, noise_in, lr_in], outputs=[status, summary_df, full_df])
    demo.queue(concurrency_count=1, max_size=2).launch(share=args.share)


if __name__ == "__main__":
    main()
