# Training Outline for Digit Classifiers

All models use the sklearn digits dataset (64-D input, 10 classes), normalized to \([0,1]\), split with `train_fraction` (default 0.8), batch size 128, optimizer Adam, learning rate from `config.yml` (default 0.01), and `epochs` (default 30). Loss: cross-entropy.

## Models
- **digital**: MLP (one or two hidden linear layers + linear output). No noise during training. Serves as accuracy ceiling.
- **phase**: MLP with linear hidden(s) + phase-encoded output layer \(w=\cos(\theta)\). Training: noiseless; inference applies phase noise `noise_std`.
- **phase_noiseaware**: Same phase encoding, but training samples phase noise std from `train_noise_list` (e.g., [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2]) each forward pass.
- **amplitude**: MLP with amplitude-coded output layer; training noiseless; inference applies multiplicative noise `noise_std`.
- **amplitude_noiseaware**: Amplitude-coded output; training samples multiplicative noise from `train_noise_list`.

## Hyperparameters (config.yml)
- `digits_demo.epochs`: total epochs per model (default 30).
- `digits_demo.lr`: learning rate (default 0.01).
- `digits_demo.noise_std`: list of noise std values swept at inference.
- `digits_demo.methods`: defines models; for phase_noiseaware, `train_noise_list` sets sampled training noises.

## Scripts
- `src/run_benchmark.py`: trains all models with the above settings, sweeps inference noise, saves `results/benchmark_digits.csv/json`.
- `src/benchmark_ui.py`: Gradio UI to run the benchmark with adjustable epochs, lr, noise list.
- `src/run_digits_demo.py`: smaller standalone sweep used in reproduce script.
