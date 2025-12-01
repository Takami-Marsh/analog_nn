# Problem Statement

## Objective
Design and evaluate an analog / wave-based neural network primitive where synaptic weights are encoded as phase or frequency shifts on a shared carrier. The target computation is an approximate multiply-accumulate (MAC) for inference workloads (linear layers, convolutions) while exploiting wave superposition for energy and bandwidth advantages demonstrated in photonic and RF analog accelerators [@shen2017deep; @feldmann2021parallel].

## Encoding hypothesis
- Represent each weight \(w_i\) as a controllable phase shift \(\phi_i = \arccos(\operatorname{clip}(w_i, -1, 1))\) applied to a carrier tone; alternatively map index \(i\) to a tone frequency \(f_i\).
- Represent each input \(x_i\) as the tone amplitude (or a small phase perturbation) at the corresponding carrier.
- Computation: the summed waveform is \(s(t)=\sum_i x_i \cos(\omega t + \phi_i) + n(t)\). IQ demodulation yields \(I \approx \sum_i x_i \cos(\phi_i)\) as the MAC estimate.
- Frequency-coded variant: assign each weight to a tone \(f_i\); apply matched filtering or FFT to extract bins that approximate \(x_i w_i\).
- Readout: coherent detection (IQ demod), correlation, or phase-locked loop (PLL) tracking as used in coherent photonics and RF receivers [@tait2017neuromorphic; @bogaerts2020programmable].

## Success metrics
- Functional accuracy: MAC mean-squared error (MSE) and downstream task accuracy (e.g., digits classification) versus an ideal digital baseline.
- Noise tolerance: SNR (dB) at which MSE < \(10^{-2}\) and classification accuracy drops <2% absolute under phase/amplitude noise [@rahman2021noise].
- Energy/bandwidth proxy: operations realized per carrier tone and reuse of a single coherent source (fewer DACs/ADCs) relative to digital MAC count, motivated by photonic tensor cores [@feldmann2021parallel].
- Scalability: how many weights/carriers can be packed before crosstalk or phase noise degrades accuracy, informed by oscillator-network and photonic scaling limits [@csaba2020coupled; @shen2017deep].
