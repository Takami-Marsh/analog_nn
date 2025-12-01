# Claim Audit

- **Phase-coded weights can approximate MAC via interference and IQ demod.** Evidence: mathematical model in `docs/04_math_model.md`; demonstrated in photonic meshes [@shen2017deep; @tait2017neuromorphic].
- **Coherent photonics achieves high-throughput convolutions.** Evidence: integrated tensor core results [@feldmann2021parallel].
- **Coupled oscillators provide an alternative compute primitive but face locking limits.** Evidence: review and Ising demonstrations [@csaba2020coupled; @wang2019ising].
- **Frequency/broadcast weighting and reservoirs scale beyond small meshes.** Evidence: broadcast-and-weight photonic networks [@tait2014broadcast]; silicon photonic reservoir computing [@vandoorne2014experimental].
- **Toy simulation shows tolerable MSE at ≥10 dB SNR.** Evidence: `results/toy_mse_vs_snr.csv`, `figures/toy_mse_vs_snr.png`.
- **Digits classifier comparison shows phase encoding tolerates ~0.1 rad jitter; amplitude coding degrades slower; noise-aware training helps.** Evidence: `results/acc_vs_noise.csv`, `figures/acc_vs_noise.png`.
- **Noise-aware or in-situ training mitigates hardware mismatch.** Evidence: physical backprop with measured transfer functions [@wright2022deep].
- **Calibration and redundancy can reduce drift/variance.** Evidence: phase tracking in photonic meshes [@bogaerts2020programmable] and noise mitigation strategies in analog accelerators [@rahman2021noise].
- **Scalable interferometer layouts reduce loss/crosstalk burden.** Evidence: compact Clements mesh design for programmable unitary matrices [@clements2016optimal].
