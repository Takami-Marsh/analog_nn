# Slide Outline (10 slides)

1. **Problem & goal** — Why analog/wave NN; target MAC and metrics.
2. **Core claim** — Phase/frequency-encoded weights enable low-energy MAC via interference.
3. **Prior work map** — Photonic tensor cores, diffractive optics, oscillator networks (citations).
4. **Architecture options** — A: coherent phase-coded; B: frequency bins; C: coupled oscillators.
5. **Chosen design** — Option A block diagram + encoding equation \( \phi_i = \arccos(w_i) \).
6. **Math model** — Signal equation, demodulation, noise terms.
7. **Simulation setup** — Config, noise parameters, pipeline (`scripts/reproduce.sh`).
8. **Results** — MSE vs SNR plot; digits accuracy vs phase noise.
9. **Noise/robustness** — Tolerable jitter/SNR, calibration hooks.
10. **Next steps** — Circuit mapping, hardware-in-loop training, prototype plan.
