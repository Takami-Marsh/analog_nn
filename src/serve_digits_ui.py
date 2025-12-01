from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def load_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class PhaseLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        train_noise: float = 0.0,
        train_noise_list: Optional[List[float]] = None,
    ):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(out_features, in_features))
        nn.init.normal_(self.theta, mean=0.0, std=0.2)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.train_noise = train_noise
        self.train_noise_list = train_noise_list or []

    def forward(
        self, x: torch.Tensor, phase_noise_std: float = 0.0, training_mode: bool = False
    ) -> torch.Tensor:
        if training_mode:
            if self.train_noise_list:
                noise_std = float(np.random.choice(self.train_noise_list))
            else:
                noise_std = self.train_noise
        else:
            noise_std = phase_noise_std
        noise = noise_std * torch.randn_like(self.theta) if noise_std > 0 else 0.0
        weight = torch.cos(self.theta + noise)
        return F.linear(x, weight, self.bias)


class AmpLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, train_noise_list: Optional[List[float]] = None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.train_noise_list = train_noise_list or []

    def forward(
        self, x: torch.Tensor, amp_noise_std: float = 0.0, training_mode: bool = False
    ) -> torch.Tensor:
        if training_mode and self.train_noise_list:
            noise_std = float(np.random.choice(self.train_noise_list))
        else:
            noise_std = amp_noise_std
        noise = noise_std * torch.randn_like(self.weight) if noise_std > 0 else 0.0
        w = self.weight * (1.0 + noise)
        return F.linear(x, w, self.bias)


def make_loaders(seed: int, train_fraction: float, batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0
    y = digits.target.astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - train_fraction, random_state=seed, stratify=y
    )
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train)
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test)
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


def train_model(
    model: nn.Module, train_loader: DataLoader, epochs: int, lr: float, device: torch.device, mode: str
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            if mode == "phase":
                logits = model(xb, training_mode=True)
            else:
                logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()


@torch.no_grad()
def predict(
    model: nn.Module, mode: str, noise_std: float, x_flat: torch.Tensor, device: torch.device
) -> Tuple[int, np.ndarray]:
    model.eval()
    if mode == "phase":
        logits = model(x_flat, phase_noise_std=noise_std)
    elif mode == "amplitude":
        logits = model(x_flat, amp_noise_std=noise_std)
    else:
        logits = model(x_flat)
    probs = F.softmax(logits, dim=1)
    top_class = int(torch.argmax(probs, dim=1).item())
    return top_class, probs.cpu().numpy().squeeze()


def preprocess_drawing(img: np.ndarray) -> np.ndarray:
    """Convert canvas RGBA/ RGB to 8x8 grayscale matching sklearn digits."""
    if img is None:
        return np.zeros((8, 8), dtype=np.float32)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    pil_img = Image.fromarray(img.astype(np.uint8))
    pil_img = pil_img.convert("L").resize((8, 8), Image.LANCZOS)
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    arr = 1.0 - arr  # invert: canvas is white background, black strokes
    return arr


def build_models(config: Dict, device: torch.device) -> Dict[str, Dict]:
    cfg = config["digits_demo"]
    train_fraction = cfg.get("train_fraction", 0.8)
    epochs = cfg.get("epochs", 10)
    lr = cfg.get("lr", 0.01)
    methods: List[Dict] = cfg.get(
        "methods",
        [
            {"name": "digital", "type": "digital"},
            {"name": "phase", "type": "phase", "train_noise": 0.0},
            {"name": "phase_noiseaware", "type": "phase", "train_noise_list": [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2]},
            {"name": "amplitude", "type": "amplitude"},
            {"name": "amplitude_noiseaware", "type": "amplitude", "train_noise_list": [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2]},
        ],
    )
    train_loader, _ = make_loaders(config.get("seed", 42), train_fraction)
    models = {}
    for method in methods:
        name = method.get("name", method.get("type", "phase"))
        mtype = method.get("type", "digital")
        if mtype == "phase":
            model = PhaseLinear(
                64,
                10,
                train_noise=float(method.get("train_noise", 0.0)),
                train_noise_list=method.get("train_noise_list"),
            )
        elif mtype == "amplitude":
            model = AmpLinear(
                64,
                10,
                train_noise_list=method.get("train_noise_list"),
            )
        else:
            model = nn.Linear(64, 10)
        model = model.to(device)
        train_model(model, train_loader, epochs=epochs, lr=lr, device=device, mode=mtype)
        models[name] = {"model": model, "mode": mtype, "train_noise": method.get("train_noise", 0.0)}
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve interactive digit analog NN UI")
    parser.add_argument("--config", type=Path, default=Path("config.yml"))
    parser.add_argument("--share", action="store_true", help="Enable gradio share")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = build_models(config, device)

    def infer(draw_img: np.ndarray, noise_values: str):
        arr = preprocess_drawing(draw_img)
        x_flat = torch.tensor(arr.flatten(), dtype=torch.float32, device=device).unsqueeze(0)
        # Parse noise list from text input (comma/space separated)
        if noise_values.strip():
            noise_list = []
            for tok in noise_values.replace(",", " ").split():
                try:
                    noise_list.append(float(tok))
                except ValueError:
                    continue
            if not noise_list:
                noise_list = [0.0, 0.05, 0.1, 0.2]
        else:
            noise_list = [0.0, 0.05, 0.1, 0.2]

        rows = []
        for model_name, meta in models.items():
            for nstd in noise_list:
                top_class, probs = predict(
                    meta["model"], meta["mode"], nstd, x_flat, device=device
                )
                for digit, p in enumerate(probs):
                    rows.append(
                        {
                            "model": model_name,
                            "noise_std": nstd,
                            "digit": str(digit),
                            "prob": float(p),
                            "pred": str(top_class),
                        }
                    )
        probs_df = pd.DataFrame(rows)
        # Best guess per model/noise
        summary = (
            probs_df.groupby(["model", "noise_std"])
            .apply(lambda g: g.loc[g["prob"].idxmax()][["digit", "prob"]])
            .reset_index()
        )
        summary = summary.rename(columns={"digit": "argmax_digit", "prob": "argmax_prob"})
        return summary, probs_df

    demo = gr.Blocks()
    with demo:
        gr.Markdown("# Analog / Phase-Coded Digit Classifier\nDraw a digit and compare models (digital, phase-encoded, noise-aware, amplitude).")
        with gr.Row():
            canvas = gr.Image(
                shape=(128, 128),
                source="canvas",
                tool="color-sketch",
                type="numpy",
                image_mode="RGB",
                invert_colors=False,
            )
            with gr.Column():
                noise_text = gr.Textbox(
                    label="Noise std list (space or comma separated)", value="0 0.05 0.1 0.2"
                )
                predict_btn = gr.Button("Predict")
                summary_table = gr.Dataframe(
                    headers=["model", "noise_std", "argmax_digit", "argmax_prob"],
                    datatype=["str", "number", "str", "number"],
                    label="Top prediction per model/noise",
                )
                probs_table = gr.Dataframe(
                    headers=["model", "noise_std", "digit", "prob", "pred"],
                    datatype=["str", "number", "str", "number", "str"],
                    label="All probabilities",
                )
        predict_btn.click(
            fn=infer,
            inputs=[canvas, noise_text],
            outputs=[summary_table, probs_table],
        )
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
