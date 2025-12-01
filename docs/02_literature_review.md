# Literature Review (Expanded)

## A) Wave / interference compute
- Shen et al. program nanophotonic interferometers to perform matrix multiplication with coherent detection and thermo-optic phase control [@shen2017deep].
- Feldmann et al. integrate a photonic tensor core using phase-change material weight banks to deliver convolutional acceleration at fJ/MAC scales [@feldmann2021parallel].
- Lin et al. demonstrate diffractive optical layers that implement a full inference pipeline passively, showing end-to-end phase-only training [@lin2018all].
- Tait et al. build silicon photonic weight banks enabling coherent summation and routing; broadcast-and-weight earlier work shows frequency-division weighting for spikes [@tait2017neuromorphic; @tait2014broadcast].
- Vandoorne et al. show reservoir computing on silicon photonics, indicating phase-sensitive nonlinearity can be harnessed for temporal tasks [@vandoorne2014experimental].
- Wright et al. use hardware-in-the-loop backpropagation through physical transfer matrices, validating co-design for noisy photonic systems [@wright2022deep].

## B) Oscillator / phase-coupling compute
- Csaba & Porod review coupled oscillators as computational primitives, detailing synchronization conditions, noise sources, and mapping to Ising/NN objectives [@csaba2020coupled].
- Hoppensteadt & Izhikevich introduce oscillatory neurocomputers where coupling strengths encode weights [@hoppensteadt1999oscillatory].
- Wang et al. realize a spin-torque-oscillator Ising machine with programmable couplings and analyze locking range limits [@wang2019ising].

## C) Mixed-signal / in-memory accelerators
- Sebastian et al. survey analog memory devices and variability impacts on MAC fidelity [@sebastian2020memory].
- Khwa et al. report a mixed-signal CNN accelerator; analog MAC arrays require calibration for gain/offset errors [@khwa2018mixed].
- Bogaerts & Chrostowski outline programmable photonic circuits, with practical phase-control and thermal drift considerations [@bogaerts2020programmable].

## D) Training and calibration methods
- Wright et al. demonstrate physical backpropagation with measured transfer matrices, reducing model-hardware mismatch [@wright2022deep].
- Tait et al. (and broadcast-and-weight) detail pilot tones, thermal tuning, and monitoring for photonic weight banks—templates for drift-aware calibration [@tait2017neuromorphic; @tait2014broadcast].
- Rahman et al. analyze algorithmic noise compensation (redundancy, retraining) in analog accelerators [@rahman2021noise].

## E) Noise & reliability
- Phase noise and coupling dispersion quantified for oscillator compute, showing jitter-driven accuracy collapse beyond locking range [@csaba2020coupled].
- Photonic drift and crosstalk documented with mitigation via feedback control in programmable meshes [@bogaerts2020programmable].
- Device variability and endurance constraints summarized for analog memories, motivating error-aware mapping [@sebastian2020memory].

## Must-read anchors (expanded to 15)
1. Shen et al., coherent nanophotonic matrix multiplication [@shen2017deep].
2. Feldmann et al., photonic tensor core with phase-change weights [@feldmann2021parallel].
3. Lin et al., diffractive optical neural networks [@lin2018all].
4. Wright et al., backprop through physical systems [@wright2022deep].
5. Tait et al., silicon photonic weight banks [@tait2017neuromorphic].
6. Tait et al., broadcast-and-weight frequency-division weighting [@tait2014broadcast].
7. Vandoorne et al., silicon photonic reservoir computing [@vandoorne2014experimental].
8. Csaba & Porod, coupled oscillator compute survey [@csaba2020coupled].
9. Hoppensteadt & Izhikevich, oscillatory neurocomputers [@hoppensteadt1999oscillatory].
10. Wang et al., oscillator-based Ising machine [@wang2019ising].
11. Sebastian et al., in-memory compute devices [@sebastian2020memory].
12. Khwa et al., mixed-signal CNN accelerator [@khwa2018mixed].
13. Bogaerts & Chrostowski, programmable photonic circuits [@bogaerts2020programmable].
14. Rahman et al., noise mitigation in analog accelerators [@rahman2021noise].
15. Clements et al., compact interferometric mesh layout enabling scalable phase control [@clements2016optimal].
