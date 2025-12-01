# Architecture Options

## Option A: Coherent superposition + IQ demod (phase-coded weights)
- **Blocks:** coherent carrier source, per-weight phase shifters (electro-optic or RF all-pass), amplitude modulators for inputs, summing waveguide/node, IQ mixer + ADC for readout [@tait2017neuromorphic; @bogaerts2020programmable].
- **Operation:** encode weight \(w_i\) via phase \(\phi_i\); modulate amplitude with \(x_i\); interference performs summation; demod extracts in-phase component approximating \(x_i w_i\) as shown in photonic meshes [@shen2017deep].
- **Scaling bottlenecks:** phase resolution vs thermal cross-talk, source linewidth-induced phase noise, DAC/ADC bandwidth, waveguide loss [@shen2017deep]; mesh depth scaling and loss can be optimized via Clements geometry [@clements2016optimal].
- **Calibration:** monitor pilot tones; run dithering to track phase drift; include phase-error term in training (hardware-aware); follow broadcast-and-weight pilot strategy for frequency reuse [@tait2014broadcast].
- **Simulatable now:** baseband waveform model with additive/phase noise; IQ demod; Monte Carlo sweeps (implemented in `src/run_toy_sim.py`).

## Option B: Frequency-coded weights (orthogonal tone bins)
- **Blocks:** tone comb generator, per-tone phase or amplitude control, linear summing network, FFT or matched-filter readout.
- **Operation:** index weight by tone \(f_i\); input modulates amplitude; orthogonality reduces cross-talk; FFT bin magnitude/phase estimates weighted sum.
- **Scaling bottlenecks:** bandwidth and comb flatness, filter leakage, phase noise causing inter-bin interference; oscillator stability [@csaba2020coupled].
- **Calibration:** periodic tone-by-tone gain/phase equalization; adaptive windowing in DSP backend.
- **Simulatable now:** discrete-time multitone synthesis plus additive/phase noise; FFT-based detection; sensitivity to comb spacing; calibration schedule similar to OFDM equalization.

## Option C: Coupled oscillator network (Kuramoto-style)
- **Blocks:** array of oscillators with tunable natural frequency/coupling, injection ports for inputs, phase measurement (digital counters or mixers).
- **Operation:** encode weights as coupling coefficients or frequency offsets; inputs perturb phases; steady-state phase differences encode inference; readout via phase order parameter [@hoppensteadt1999oscillatory; @wang2019ising].
- **Scaling bottlenecks:** frequency crowding, locking range limits, coupling graph sparsity, start-up time [@csaba2020coupled].
- **Calibration:** locking-range characterization; adaptive biasing to offset drift; training-in-the-loop to learn robust couplings.
- **Simulatable now:** Kuramoto ODEs with noise; Monte Carlo for lock probability; parameter sweeps for coupling strength and coupling topology.

## Decision heuristic
- Phase-coded coherent superposition (Option A) offers the cleanest link to conventional MACs and can be prototyped with off-the-shelf IQ demodulation.
- Frequency-coded (Option B) is robust to static phase drift but bandwidth-heavy.
- Coupled oscillators (Option C) naturally implement recurrent or Ising-like objectives; useful as an alternative baseline rather than primary MAC engine.
