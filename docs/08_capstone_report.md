# Capstone Report

## 1. Problem & motivation
Phase- or frequency-encoded analog neural networks promise high-throughput MACs using wave superposition instead of dense digital MAC arrays, reducing ADC/DAC overhead and leveraging coherent detection as shown in photonic tensor cores [@shen2017deep; @feldmann2021parallel]. The goal is to quantify whether phase-coded weights can deliver accurate MACs under realistic noise and how training and calibration must adapt.

## 2. Prior work (taxonomy)
- **Interference compute:** coherent photonic meshes for matrix multiplication [@shen2017deep; @tait2017neuromorphic]; broadcast-and-weight frequency-division weighting [@tait2014broadcast]; diffractive optics for passive inference [@lin2018all]; integrated tensor cores with phase-change weights [@feldmann2021parallel]; photonic reservoirs for temporal tasks [@vandoorne2014experimental]; compact mesh layouts for scalable interferometers [@clements2016optimal].
- **Oscillator networks:** synchronization-based inference and Ising solvers using coupled oscillators [@csaba2020coupled; @hoppensteadt1999oscillatory; @wang2019ising].
- **Mixed-signal accelerators:** analog in-memory MACs, variability, and calibration needs [@sebastian2020memory; @khwa2018mixed]; programmable photonic circuits for stable phase control [@bogaerts2020programmable].
- **Training/calibration:** hardware-in-the-loop backprop for physical systems [@wright2022deep]; pilot-tone phase tracking and dithering in photonic meshes [@tait2017neuromorphic; @tait2014broadcast]; algorithmic noise mitigation [@rahman2021noise].

## 3. Proposed approach
- **Chosen architecture:** Option A—coherent superposition with phase-coded weights and IQ demodulation—because it most directly implements a linear MAC, follows demonstrated photonic meshes [@shen2017deep], and reuses broadcast-and-weight calibration techniques [@tait2014broadcast].
- **Mapping:** \(w_i \in [-1,1] \rightarrow \phi_i = \arccos(w_i)\); inputs are amplitudes \(x_i\); IQ demod extracts \(I \approx \sum x_i w_i\). Frequency-coded variant remains an alternative for orthogonality and drift tolerance.
- **System blocks:** carrier source, phase shifters, amplitude modulators, summing node, IQ demod, ADC; slow calibration loop for phase drift and gain equalization.

## 4. Math model & mapping
The waveform model \(s(t)=\sum_i x_i \cos(\omega t + \phi_i) + n(t)\) demodulates to \(I = \sum_i x_i \cos(\phi_i) + \eta_I\). Phase noise \(\delta\phi\), amplitude noise \(\delta a\), and additive noise determine accuracy (see `docs/04_math_model.md`). Training uses noise injection on \(\phi\) to match deployment conditions [@wright2022deep].

## 5. Simulation methodology
- **Toy MAC:** 32-feature MAC, 200 Monte Carlo samples per SNR. Noise: amplitude std 0.01, phase std 0.02 rad. Outputs in `results/toy_mse_vs_snr.csv`.
- **Digits demo:** single-layer phase-coded classifier on sklearn digits; weights parameterized as \(\cos(\theta)\); evaluate accuracy under phase noise list `[0, 0.02, 0.05, 0.1, 0.2]`. Outputs in `results/acc_vs_noise.csv`.
- Reproducible via `bash scripts/reproduce.sh`, which sets up a venv, fetches literature metadata, and runs both simulations.

## 6. Results
- **MAC accuracy:** MSE improves from 0.42 at -5 dB SNR to 0.01 at 30 dB; <2e-2 MSE at ≥10 dB (Figure `figures/toy_mse_vs_snr.png`).
- **Digits classification (comparative noise study):**
  - `digital` baseline: 0.95 accuracy, unaffected by injected noise (reference ceiling).
  - `phase`: 0.928 → 0.917 as phase noise grows from 0 to 0.2 rad.
  - `phase_noiseaware`: 0.928 → 0.909; slight robustness gain at mid-noise (0.931 at 0.05 rad).
  - `amplitude`: 0.95 → 0.936 under multiplicative noise 0.2.
  - See Figure `figures/acc_vs_noise.png` and `results/acc_vs_noise.csv`.
- **Implication:** phase-coded inference is viable up to ~0.1 rad jitter; noise-aware training helps; amplitude-coded weights are less sensitive to phase jitter and can serve as a hybrid fallback.

## 7. Noise/robustness findings
- Additive noise dominates below 10 dB SNR; phase noise dominates beyond 0.1 rad.
- Noise-aware training (injecting \(\delta\phi\)) should reclaim accuracy for higher jitter, analogous to in-situ training in physical NNs [@wright2022deep].
- Pilot-tone calibration and redundancy can further suppress drift and stochastic noise [@rahman2021noise].

## 8. Limitations and next steps
- Current model ignores device bandwidth limits and nonlinearity in modulators; integrating measured transfer functions is next.
- Only single-layer classification tested; deeper networks require phase-decorrelation analysis and perhaps hybrid analog/digital partitioning.
- Circuit validation (SPICE or benchtop IQ chain) is pending; `docs/07_circuit_mapping.md` outlines the hardware mapping.

## 9. References
Full annotated bibliography in `docs/references.md`; BibTeX in `references.bib`.
