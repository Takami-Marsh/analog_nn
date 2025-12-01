# Circuit Mapping (concept)

## Phase-coded coherent sum (Option A)
- **Phase shift elements:** thermo-optic or electro-optic phase shifters (silicon photonics) or RF all-pass networks; control voltage maps to \(\phi_i\) [@bogaerts2020programmable].
- **Amplitude modulation:** Mach–Zehnder modulators or IQ mixers to encode \(x_i\).
- **Summing node:** multimode interference (MMI) coupler or RF power combiner provides linear superposition.
- **Readout:** balanced photodiodes + IQ mixer; ADC at baseband.
- **Calibration:** pilot tones at known \(\phi\) for phase drift tracking; dithering of heater currents as in photonic weight banks [@tait2017neuromorphic].

## Frequency-coded weights (Option B)
- **Tone comb generation:** PLL + divider bank or optical frequency comb.
- **Weight control:** per-tone phase or amplitude shift using varactor banks or micro-ring resonators.
- **Readout:** FFT engine or bank of narrowband filters; AGC loop to equalize tone amplitudes.
- **Concern:** inter-bin leakage from phase noise; requires sufficient tone spacing and low-jitter source [@csaba2020coupled]; broadcast-and-weight shows shared-source stability advantages [@tait2014broadcast].

## Coupled oscillators (Option C)
- **Elements:** ring/VCO/LC oscillators with tunable natural frequency; coupling via resistive/capacitive or mutual inductive links.
- **Programming:** bias currents or coupling capacitors encode weights/couplings [@hoppensteadt1999oscillatory; @wang2019ising].
- **Readout:** phase detectors or time-to-digital converters compute order parameters.
- **Stability:** locking range sets usable coupling magnitude; start-up time dictates latency; device mismatch requires per-oscillator trimming [@csaba2020coupled].

## Practical prototype path
1. Build benchtop emulation: RF source → IQ modulator → digitally controlled phase shifters (vector network analyzer or SDR) → IQ demod → Python processing.
2. Validate mapping between digital weight and measured \(\phi\); characterize noise \((\sigma_\phi, \sigma_a)\).
3. Iterate training with measured noise statistics using the provided simulation scripts.
