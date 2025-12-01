from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


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
        noise_std = 0.0
        if training_mode:
            if self.train_noise_list:
                noise_std = float(np.random.choice(self.train_noise_list))
            else:
                noise_std = self.train_noise
        else:
            noise_std = phase_noise_std
        if noise_std > 0:
            noise = noise_std * torch.randn_like(self.theta)
        else:
            noise = 0.0
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


class AnalogMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        mode: str,
        train_noise: float = 0.0,
        train_noise_list: Optional[List[float]] = None,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.hidden = nn.Sequential(*layers) if layers else nn.Identity()
        self.mode = mode
        if mode == "phase":
            self.out = PhaseLinear(prev, 10, train_noise=train_noise, train_noise_list=train_noise_list)
        elif mode == "amplitude":
            self.out = AmpLinear(prev, 10, train_noise_list=train_noise_list)
        else:
            self.out = nn.Linear(prev, 10)

    def forward(
        self,
        x: torch.Tensor,
        phase_noise_std: float = 0.0,
        amp_noise_std: float = 0.0,
        training_mode: bool = False,
    ) -> torch.Tensor:
        h = self.hidden(x)
        if self.mode == "phase":
            return self.out(h, phase_noise_std=phase_noise_std, training_mode=training_mode)
        elif self.mode == "amplitude":
            return self.out(h, amp_noise_std=amp_noise_std, training_mode=training_mode)
        else:
            return self.out(h)


def make_loaders(
    seed: int, train_fraction: float, batch_size: int, dataset: str, input_dim: int
) -> Tuple[DataLoader, DataLoader]:
    if dataset.lower() == "mnist":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), transforms.Lambda(lambda t: t.view(-1))]
        )
        train_set = datasets.MNIST("data", train=True, download=True, transform=transform)
        test_set = datasets.MNIST("data", train=False, download=True, transform=transform)
        # Optionally split train_fraction
        n_train = int(len(train_set) * train_fraction)
        n_val = len(train_set) - n_train
        train_subset, _ = torch.utils.data.random_split(
            train_set, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
        )
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader
    else:
        # fallback to sklearn digits
        from sklearn.datasets import load_digits

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
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    mode: str,
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            if mode in ("phase", "amplitude"):
                logits = model(xb, training_mode=True)
            else:
                logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    mode: str,
    noise_std: float,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for xb, yb in data_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        if mode == "phase":
            logits = model(xb, phase_noise_std=noise_std)
        elif mode == "amplitude":
            logits = model(xb, amp_noise_std=noise_std)
        else:
            logits = model(xb)
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == yb).sum().item())
        total += yb.size(0)
    return correct / total if total > 0 else 0.0


def build_model(method: Dict, device: torch.device, input_dim: int, hidden_dims: List[int]) -> Tuple[nn.Module, str]:
    mtype = method.get("type", "digital")
    model = AnalogMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        mode=mtype,
        train_noise=float(method.get("train_noise", 0.0)),
        train_noise_list=method.get("train_noise_list"),
    )
    return model.to(device), mtype


def run_experiment(config: Dict) -> pd.DataFrame:
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = config["digits_demo"]
    train_fraction = cfg.get("train_fraction", 0.8)
    epochs = cfg.get("epochs", 8)
    lr = cfg.get("lr", 0.01)
    dataset = cfg.get("dataset", "digits")
    input_dim = cfg.get("input_dim", 64)
    hidden_dims = cfg.get("hidden_dims", [64])
    noise_std_list: List[float] = cfg.get("noise_std", [0.0, 0.05, 0.1])
    methods: List[Dict] = cfg.get("methods", [{"name": "phase", "type": "phase"}])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = make_loaders(
        seed, train_fraction, batch_size=128, dataset=dataset, input_dim=input_dim
    )

    records = []
    for method in methods:
        name = method.get("name", method.get("type", "phase"))
        model, mode = build_model(method, device, input_dim=input_dim, hidden_dims=hidden_dims)
        train_model(model, train_loader, epochs=epochs, lr=lr, device=device, mode=mode)
        for noise_std in noise_std_list:
            acc = evaluate(model, test_loader, device, mode=mode, noise_std=noise_std)
            records.append(
                {
                    "method": name,
                    "type": mode,
                    "noise_std": float(noise_std),
                    "train_noise": float(method.get("train_noise", 0.0)),
                    "accuracy": float(acc),
                }
            )
    return pd.DataFrame.from_records(records)


def plot_results(df: pd.DataFrame, fig_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    for method, group in df.groupby("method"):
        plt.plot(group["noise_std"], group["accuracy"], marker="o", label=method)
    plt.xlabel("Noise std (phase or amplitude)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()


def save_results(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Digits demo with phase-coded weights")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--results", type=Path, default=Path("results/acc_vs_noise.csv")
    )
    parser.add_argument(
        "--figure", type=Path, default=Path("figures/acc_vs_noise.png")
    )
    args = parser.parse_args()
    config = load_config(args.config)
    cfg = config.get("digits_demo", {})
    if not cfg.get("enabled", True):
        print("digits_demo.disabled")
        return
    df = run_experiment(config)
    save_results(df, args.results)
    plot_results(df, args.figure)
    print(df)


if __name__ == "__main__":
    main()
