import argparse
import threading
from pathlib import Path
from typing import Dict, List

import gradio as gr
import pandas as pd
import yaml
import torch

from run_benchmark import get_device, load_config, run_benchmark
from run_benchmark_fashion import run_benchmark_fashion


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio UI to run digit benchmark")
    parser.add_argument("--config", type=Path, default=Path("config.yml"))
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    base_config = load_config(args.config)
    device = get_device()
    print(f"[benchmark_ui] Using device: {device}")
    stop_event = threading.Event()

    variants = {
        "digits_demo": "MNIST baseline (digits_demo)",
        "fashion_complex": "Fashion-MNIST",
        "kmnist_benchmark": "Kuzushiji-MNIST",
        "emnist_letters_benchmark": "EMNIST Letters",
        "cifar10_flat_benchmark": "CIFAR-10 (flattened)",
    }

    def available_devices() -> List[str]:
        devices: List[str] = []
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            devices.append("mps")
        if torch.cuda.is_available():
            devices.append("cuda")
        devices.append("cpu")
        return devices

    def defaults_for_variant(benchmark_key: str):
        cfg = base_config.get(benchmark_key, {})
        fallback_cfg = base_config.get("digits_demo", {})
        epochs_default = cfg.get("epochs", fallback_cfg.get("epochs", 30))
        lr_default = cfg.get("lr", fallback_cfg.get("lr", 0.01))
        noise_default = " ".join(map(str, cfg.get("noise_std", fallback_cfg.get("noise_std", []))))
        return epochs_default, lr_default, noise_default

    def run_with_override(benchmark_key: str, epochs: int, noise_text: str, lr: float, device_choice: str):
        progress = gr.Progress(track_tqdm=True)
        stop_event.clear()
        cfg = base_config.copy()
        cfg[benchmark_key] = cfg.get(benchmark_key, {}).copy()
        cfg[benchmark_key]["epochs"] = epochs
        cfg[benchmark_key]["lr"] = lr
        if noise_text.strip():
            try:
                noise_list: List[float] = [float(tok) for tok in noise_text.replace(",", " ").split()]
                cfg[benchmark_key]["noise_std"] = noise_list
            except ValueError:
                pass
        progress(0, desc="Starting benchmark...")
        device_used = torch.device(device_choice)
        if benchmark_key == "digits_demo":
            df = run_benchmark(cfg, device=device_used, stop_event=stop_event)
        else:
            df = run_benchmark_fashion(cfg, config_key=benchmark_key, device=device_used, stop_event=stop_event)
        if df.empty or "model" not in df.columns:
            progress(1.0, desc="Stopped")
            return "Stopped with no results (likely due to stop request).", pd.DataFrame(), df
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

    def request_stop():
        stop_event.set()
        return "Stop requested... finishing current batch/epoch."

    with gr.Blocks() as demo:
        gr.Markdown(
            "# Benchmark: Analog / Digital Image Classifiers\nRuns training + evaluation with configurable epochs, noise list, and dataset selection."
        )
        with gr.Row():
            benchmark_selector = gr.Dropdown(
                choices=list(variants.keys()),
                value="digits_demo",
                label="Benchmark variant",
                info="Select which benchmark config to run",
            )
            epochs_default, lr_default, noise_default = defaults_for_variant("digits_demo")
            epochs_in = gr.Slider(5, 80, value=epochs_default, step=1, label="Epochs")
            lr_in = gr.Slider(0.0005, 0.05, value=lr_default, step=0.0005, label="Learning rate")
            noise_in = gr.Textbox(label="Noise std list", value=noise_default)
            device_in = gr.Dropdown(
                choices=available_devices(),
                value=str(device),
                label="Device",
                info="Only available devices are shown",
            )
            run_btn = gr.Button("Run benchmark")
            stop_btn = gr.Button("Stop benchmark", variant="stop")
        status = gr.Textbox(label="Status", interactive=False)
        summary_df = gr.Dataframe(label="Summary (mean/min/max accuracy by model)")
        full_df = gr.Dataframe(label="Full results (per noise)")
        benchmark_selector.change(fn=defaults_for_variant, inputs=benchmark_selector, outputs=[epochs_in, lr_in, noise_in])
        stop_btn.click(fn=request_stop, outputs=status)
        run_btn.click(
            fn=run_with_override,
            inputs=[benchmark_selector, epochs_in, noise_in, lr_in, device_in],
            outputs=[status, summary_df, full_df],
        )
    demo.queue(concurrency_count=3, max_size=4).launch(share=args.share)


if __name__ == "__main__":
    main()
