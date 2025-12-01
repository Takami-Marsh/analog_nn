# Mathematical Model

## Signal model (phase-coded)
Given carrier frequency \(\omega\) and \(N\) inputs:
\[
s(t) = \sum_{i=1}^{N} A_i(x_i) \cos(\omega t + \phi_i(w_i)) + n(t)
\]
with \(A_i(x_i) = x_i\) (real-valued amplitude) and \(\phi_i(w_i) = \arccos(\operatorname{clip}(w_i, -1, 1))\) so that \(\cos(\phi_i) = w_i\).

IQ demodulation with reference \(\cos(\omega t)\) and \(\sin(\omega t)\) yields:
\[
I = \frac{2}{T}\int_0^{T} s(t) \cos(\omega t)\,dt \approx \sum_i x_i \cos(\phi_i) + \eta_I
\]
\[
Q = \frac{2}{T}\int_0^{T} s(t) \sin(\omega t)\,dt \approx \sum_i x_i \sin(\phi_i) + \eta_Q
\]
The MAC estimate is \(\hat{y}=I\) for real weights; magnitude \(|I + jQ|\) can capture amplitude errors. A similar derivation holds for frequency bins with orthogonal tones and FFT-based integration.

## Noise terms
- Additive receiver noise \(n(t)\) modeled as white Gaussian; variance selected for a given SNR.
- Phase noise: \(\phi_i \leftarrow \phi_i + \delta\phi_i\), \(\delta\phi_i \sim \mathcal{N}(0, \sigma_\phi^2)\) [@csaba2020coupled].
- Amplitude noise: \(A_i \leftarrow A_i (1+\delta a_i)\) with \(\delta a_i \sim \mathcal{N}(0, \sigma_a^2)\) [@rahman2021noise].

## Training implications
- **Offline mapping:** Train digital weights \(w_i\in[-1,1]\); map to \(\phi_i=\arccos(w_i)\). This preserves the real MAC in the noiseless limit and constrains hardware to phase shifts.
- **Noise-aware loss:** During training, inject \(\delta\phi_i\) and \(\delta a_i\) so gradients account for expected jitter and amplitude variation, following hardware-in-the-loop methods [@wright2022deep].
- **Calibration-aware bias:** If systematic phase offsets \(\bar{\phi}_i\) are measured, update mapping to \(\phi_i=\arccos(w_i)-\bar{\phi}_i\); residual error is tracked by periodic test vectors as used in photonic weight banks [@tait2017neuromorphic].

## Readout and scaling
- Readout bandwidth must cover at least the carrier plus modulation sidebands; FFT bin spacing must exceed phase-noise-induced linewidth for frequency-coded weights [@bogaerts2020programmable].
- Summation linearity holds while phase shifters stay within small-signal regime and coupling terms remain independent; thermal cross-talk or parasitic coupling can be modeled as correlated phase perturbations [@shen2017deep].
