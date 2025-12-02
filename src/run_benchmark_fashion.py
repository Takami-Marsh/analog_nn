from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import threading

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from run_benchmark import build_model, eval_model, get_device, load_config, save_outputs, train_model


def make_loaders_fashion(
    seed: int, train_fraction: float, batch_size: int, dataset: str, input_dim: int
) -> Tuple[DataLoader, DataLoader]:
    ds_name = dataset.lower()
    if ds_name in {"fashion-mnist", "fashion_mnist", "fashion"}:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), transforms.Lambda(lambda t: t.view(-1))]
        )
        train_set = datasets.FashionMNIST("data", train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST("data", train=False, download=True, transform=transform)
        n_train = int(len(train_set) * train_fraction)
        n_val = len(train_set) - n_train
        train_subset, _ = torch.utils.data.random_split(
            train_set, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
        )
        return (
            DataLoader(train_subset, batch_size=batch_size, shuffle=True),
            DataLoader(test_set, batch_size=batch_size, shuffle=False),
        )
    elif ds_name == "kmnist":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), transforms.Lambda(lambda t: t.view(-1))]
        )
        train_set = datasets.KMNIST("data", train=True, download=True, transform=transform)
        test_set = datasets.KMNIST("data", train=False, download=True, transform=transform)
        n_train = int(len(train_set) * train_fraction)
        n_val = len(train_set) - n_train
        train_subset, _ = torch.utils.data.random_split(
            train_set, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
        )
        return (
            DataLoader(train_subset, batch_size=batch_size, shuffle=True),
            DataLoader(test_set, batch_size=batch_size, shuffle=False),
        )
    elif ds_name in {"emnist-letters", "emnist_letters", "emnist_letters_split"}:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), transforms.Lambda(lambda t: t.view(-1))]
        )
        train_set = datasets.EMNIST(
            "data", split="letters", train=True, download=True, transform=transform, target_transform=lambda y: y - 1
        )
        test_set = datasets.EMNIST(
            "data", split="letters", train=False, download=True, transform=transform, target_transform=lambda y: y - 1
        )
        n_train = int(len(train_set) * train_fraction)
        n_val = len(train_set) - n_train
        train_subset, _ = torch.utils.data.random_split(
            train_set, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
        )
        return (
            DataLoader(train_subset, batch_size=batch_size, shuffle=True),
            DataLoader(test_set, batch_size=batch_size, shuffle=False),
        )
    elif ds_name == "cifar10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
                transforms.Lambda(lambda t: t.view(-1)),
            ]
        )
        train_set = datasets.CIFAR10("data", train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10("data", train=False, download=True, transform=transform)
        n_train = int(len(train_set) * train_fraction)
        n_val = len(train_set) - n_train
        train_subset, _ = torch.utils.data.random_split(
            train_set, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
        )
        return (
            DataLoader(train_subset, batch_size=batch_size, shuffle=True),
            DataLoader(test_set, batch_size=batch_size, shuffle=False),
        )
    elif ds_name == "mnist":
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
        # fallback to sklearn digits
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


def run_benchmark_fashion(
    config: Dict,
    config_key: str = "fashion_complex",
    device: Optional[torch.device] = None,
    stop_event: Optional[threading.Event] = None,
) -> pd.DataFrame:
    should_stop = stop_event.is_set if stop_event else (lambda: False)
    if config_key not in config:
        raise KeyError(f"Config key '{config_key}' not found in config file")
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = config[config_key]
    epochs = cfg.get("epochs", 50)
    lr = cfg.get("lr", 0.003)
    dataset = cfg.get("dataset", "fashion-mnist")
    input_dim = cfg.get("input_dim", 784)
    hidden_dims = cfg.get("hidden_dims", [512, 256, 128])
    batch_size = cfg.get("batch_size", 128)
    num_classes = cfg.get("num_classes", 10)
    noise_list: List[float] = cfg.get("noise_std", [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4])
    methods: List[Dict] = cfg.get("methods", [{"name": "digital", "type": "digital"}])
    device = torch.device(device) if device else get_device()

    train_loader, test_loader = make_loaders_fashion(
        seed, cfg.get("train_fraction", 0.8), batch_size=batch_size, dataset=dataset, input_dim=input_dim
    )

    rows = []
    for method in tqdm(methods, desc="methods"):
        if should_stop():
            break
        name = method.get("name", method.get("type", "phase"))
        model, mtype = build_model(
            method, device, input_dim=input_dim, hidden_dims=hidden_dims, num_classes=num_classes
        )
        train_model(model, train_loader, epochs=epochs, lr=lr, device=device, mode=mtype, stop_check=should_stop)
        if should_stop():
            break
        for nstd in tqdm(noise_list, desc=f"noise {name}", leave=False):
            acc = eval_model(model, test_loader, device=device, mode=mtype, noise_std=nstd, stop_check=should_stop)
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
                    "dataset": dataset,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark analog/digital classifiers on Fashion-MNIST and other datasets")
    parser.add_argument("--config", type=Path, default=Path("config.yml"))
    parser.add_argument(
        "--config-key",
        type=str,
        default="fashion_complex",
        help="Which config block to use (e.g., fashion_complex, kmnist_benchmark, emnist_letters_benchmark, cifar10_flat_benchmark).",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Output CSV path; defaults based on config key.")
    parser.add_argument("--json", type=Path, default=None, help="Output JSON path; defaults based on config key.")
    args = parser.parse_args()
    config = load_config(args.config)
    df = run_benchmark_fashion(config, config_key=args.config_key)

    default_csv = Path(f"results/benchmark_{args.config_key}.csv")
    default_json = Path(f"results/benchmark_{args.config_key}.json")
    csv_path = args.csv or default_csv
    json_path = args.json or default_json

    save_outputs(df, csv_path, json_path)
    print(df)


if __name__ == "__main__":
    main()
