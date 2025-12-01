import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def load_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def simulate_weighted_sum(
    x: np.ndarray,
    phi: np.ndarray,
    carrier_freq: float,
    phase_noise_std: float,
    amplitude_noise_std: float,
    snr_db: float,
    rng: np.random.Generator,
) -> float:
    """Simulate a single weighted sum using a phase-encoded carrier."""
    num_points = 256
    t = np.linspace(0, 2 * np.pi, num_points)
    omega_t = carrier_freq * t
    signal = np.zeros_like(t)
    for amp, phi_i in zip(x, phi):
        noisy_amp = amp * (1 + rng.normal(0, amplitude_noise_std))
        noisy_phi = phi_i + rng.normal(0, phase_noise_std)
        signal += noisy_amp * np.cos(omega_t + noisy_phi)
    signal_power = np.mean(signal**2)
    noise_power = signal_power / (10 ** (snr_db / 10)) if snr_db is not None else 0.0
    noise_sigma = np.sqrt(noise_power)
    noisy_signal = signal + rng.normal(0, noise_sigma, size=t.shape)
    # IQ demodulation to recover the in-phase component
    i_corr = (2 / num_points) * np.sum(noisy_signal * np.cos(omega_t))
    q_corr = (2 / num_points) * np.sum(noisy_signal * np.sin(omega_t))
    magnitude = np.hypot(i_corr, q_corr)
    # For a real-weighted MAC we take the in-phase term as the estimator
    return i_corr, magnitude


def run_experiment(config: Dict) -> pd.DataFrame:
    seed = config.get("seed", 42)
    rng = np.random.default_rng(seed)
    toy_cfg = config["toy_sim"]
    n_samples = toy_cfg.get("n_samples", 200)
    n_features = toy_cfg.get("n_features", 32)
    carrier_freq = toy_cfg.get("carrier_freq", 1.0)
    phase_noise_std = toy_cfg.get("phase_noise_std", 0.02)
    amplitude_noise_std = toy_cfg.get("amplitude_noise_std", 0.01)
    snr_db_list: List[float] = toy_cfg.get("snr_db_list", [0, 5, 10, 20])
    rng = np.random.default_rng(seed)
    # Generate weights and phases: encode weight in cosine of phase
    w_true = rng.uniform(-0.9, 0.9, size=n_features)
    phi = np.arccos(w_true)
    records = []
    for snr_db in snr_db_list:
        mse_list = []
        corr_list = []
        mag_list = []
        for _ in range(n_samples):
            x = rng.normal(0, 1, size=n_features)
            y_true = float(np.dot(x, w_true))
            i_corr, magnitude = simulate_weighted_sum(
                x,
                phi,
                carrier_freq,
                phase_noise_std,
                amplitude_noise_std,
                snr_db,
                rng,
            )
            mse_list.append((i_corr - y_true) ** 2)
            mag_list.append(magnitude)
            corr_list.append(i_corr * y_true)
        mse = float(np.mean(mse_list))
        corr = float(np.mean(corr_list) / (np.sqrt(np.mean(np.square(w_true))) + 1e-9))
        records.append(
            {
                "snr_db": snr_db,
                "mse": mse,
                "mean_corr": corr,
                "mean_magnitude": float(np.mean(mag_list)),
            }
        )
    return pd.DataFrame.from_records(records)


def plot_results(df: pd.DataFrame, fig_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(df["snr_db"], df["mse"], marker="o", label="MSE vs SNR")
    plt.xlabel("SNR (dB)")
    plt.ylabel("MSE")
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
    parser = argparse.ArgumentParser(description="Run toy analog MAC simulation")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    parser.add_argument(
        "--results", type=Path, default=Path("results/toy_mse_vs_snr.csv")
    )
    parser.add_argument(
        "--figure", type=Path, default=Path("figures/toy_mse_vs_snr.png")
    )
    args = parser.parse_args()
    config = load_config(args.config)
    df = run_experiment(config)
    save_results(df, args.results)
    plot_results(df, args.figure)
    summary = df.to_dict(orient="records")
    print(json.dumps({"results": summary}, indent=2))


if __name__ == "__main__":
    main()
