from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
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
        train_noise_list: List[float] | None = None,
    ):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(out_features, in_features))
        nn.init.normal_(self.theta, mean=0.0, std=0.2)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.train_noise = train_noise
        self.train_noise_list = train_noise_list or []

    def forward(self, x: torch.Tensor, phase_noise_std: float = 0.0, training_mode: bool = False):
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

    def forward(self, x: torch.Tensor, amp_noise_std: float = 0.0, training_mode: bool = False):
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


def make_loaders(seed: int, train_fraction: float, batch_size: int, dataset: str, input_dim: int) -> Tuple[DataLoader, DataLoader]:
    if dataset.lower() == "mnist":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), transforms.Lambda(lambda t: t.view(-1))]
        )
        train_set = datasets.MNIST("data", train=True, download=True, transform=transform)
        test_set = datasets.MNIST("data", train=False, download=True, transform=transform)
        n_train = int(len(train_set) * train_fraction)
        n_val = len(train_set) - n_train
        train_subset, _ = torch.utils.data.random_split(
            train_set, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
        )
        return (
            DataLoader(train_subset, batch_size=batch_size, shuffle=True),
            DataLoader(test_set, batch_size=batch_size, shuffle=False),
        )
    else:
        from sklearn.datasets import load_digits

        digits = load_digits()
        X = digits.data.astype(np.float32) / 16.0
        y = digits.target.astype(np.int64)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1 - train_fraction, random_state=seed, stratify=y
        )
        train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        )


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


def train_model(model: nn.Module, loader: DataLoader, epochs: int, lr: float, device: torch.device, mode: str):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in tqdm(range(epochs), desc="train", leave=False):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device, dtype=torch.float32)
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
def eval_model(model: nn.Module, loader: DataLoader, device: torch.device, mode: str, noise_std: float) -> float:
    model.eval()
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device, dtype=torch.float32)
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
    return correct / total if total else 0.0


def run_benchmark(config: Dict) -> pd.DataFrame:
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = config["digits_demo"]
    epochs = cfg.get("epochs", 30)
    lr = cfg.get("lr", 0.01)
    dataset = cfg.get("dataset", "digits")
    input_dim = cfg.get("input_dim", 64)
    hidden_dims = cfg.get("hidden_dims", [64])
    noise_list: List[float] = cfg.get("noise_std", [0.0, 0.05, 0.1, 0.2])
    methods: List[Dict] = cfg.get("methods", [{"name": "digital", "type": "digital"}])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = make_loaders(seed, cfg.get("train_fraction", 0.8), batch_size=128, dataset=dataset, input_dim=input_dim)

    rows = []
    for method in tqdm(methods, desc="methods"):
        name = method.get("name", method.get("type", "phase"))
        model, mtype = build_model(method, device, input_dim=input_dim, hidden_dims=hidden_dims)
        train_model(model, train_loader, epochs=epochs, lr=lr, device=device, mode=mtype)
        for nstd in tqdm(noise_list, desc=f"noise {name}", leave=False):
            acc = eval_model(model, test_loader, device=device, mode=mtype, noise_std=nstd)
            rows.append(
                {
                    "model": name,
                    "type": mtype,
                    "train_noise": float(method.get("train_noise", 0.0)),
                    "noise_std": float(nstd),
                    "accuracy": float(acc),
                    "epochs": epochs,
                    "lr": lr,
                    "seed": seed,
                }
            )
    return pd.DataFrame(rows)


def save_outputs(df: pd.DataFrame, csv_path: Path, json_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(df.to_dict(orient="records"), indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark analog/digital digit classifiers across noise")
    parser.add_argument("--config", type=Path, default=Path("config.yml"))
    parser.add_argument("--csv", type=Path, default=Path("results/benchmark_digits.csv"))
    parser.add_argument("--json", type=Path, default=Path("results/benchmark_digits.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    df = run_benchmark(config)
    save_outputs(df, args.csv, args.json)
    print(df)


if __name__ == "__main__":
    main()
