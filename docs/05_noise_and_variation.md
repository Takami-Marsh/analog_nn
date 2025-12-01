# Noise and Variation

## Noise taxonomy
- **Additive receiver noise:** thermal/shot noise at the detector or mixer; reduces SNR and raises demodulation MSE [@rahman2021noise].
- **Phase noise / jitter:** source linewidth or oscillator flicker; manifests as random \(\delta\phi\) that directly perturbs effective weight [@csaba2020coupled].
- **Amplitude noise:** modulator gain fluctuation or laser intensity noise; perturbs \(A_i(x_i)\).
- **Component mismatch:** static offsets in phase shifters or coupling coefficients; causes bias that accumulates with network depth [@sebastian2020memory].
- **Temperature drift:** slow variations causing correlated phase shifts across weights, observed in photonic meshes [@bogaerts2020programmable].
- **Quantization:** finite-resolution phase shifters or LUT-based frequency placement.

## Sensitivity analysis plan
- Sweep SNR vs MAC MSE using the toy waveform simulator (`src/run_toy_sim.py`).
- Sweep phase-noise standard deviation vs classification accuracy on digits (`src/run_digits_demo.py`).
- Evaluate mismatch by adding static offsets to \(\phi_i\) and re-running MAC MSE; track bias vs variance.
- Record tolerance thresholds: max \(\sigma_\phi\) and min SNR to maintain <2% accuracy drop.

## Monte Carlo setup
- \(N=200\) samples per SNR/phase-noise setting with fixed RNG seed for reproducibility.
- For each configuration: compute mean and 95% confidence interval of MSE/accuracy.
- Reported outputs:
  - `results/toy_mse_vs_snr.csv` and `figures/toy_mse_vs_snr.png`
  - `results/acc_vs_noise.csv` and `figures/acc_vs_noise.png`

## Calibration hooks
- Pilot-tone based phase tracking (Option A) to remove slow drift [@tait2017neuromorphic].
- Retraining or fine-tuning with measured \(\delta\phi\) distribution to compensate stochastic noise [@wright2022deep].
- Redundancy/averaging (duplicate carriers or oscillators) to lower effective variance [@rahman2021noise].
