# Training Outline for Image Classifier Benchmarks

All benchmarks use cross-entropy loss, Adam optimizer, batch size (default 128), and the configured noise sweeps. Variants differ by dataset and MLP depth:
- **digits_demo** (default): MNIST (28×28 → 784-D), `epochs` 40, `lr` 0.005.
- **fashion_complex**: Fashion-MNIST (28×28 → 784-D) with hidden dims `[512, 256, 128]`, `epochs` 50, `lr` 0.003.
- **kmnist_benchmark**: Kuzushiji-MNIST (28×28 → 784-D), `epochs` 45, `lr` 0.0035.
- **emnist_letters_benchmark**: EMNIST Letters (28×28 → 784-D, 26 classes), `epochs` 55, `lr` 0.003.
- **cifar10_flat_benchmark**: CIFAR-10 flattened (32×32×3 → 3072-D), `epochs` 65, `lr` 0.0025.

## Models
- **digital**: MLP (one or two hidden linear layers + linear output). No noise during training. Serves as accuracy ceiling.
- **phase**: MLP with linear hidden(s) + phase-encoded output layer \(w=\cos(\theta)\). Training: noiseless; inference applies phase noise `noise_std`.
- **phase_noiseaware**: Same phase encoding, but training samples phase noise std from `train_noise_list` (e.g., [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2]) each forward pass.
- **amplitude**: MLP with amplitude-coded output layer; training noiseless; inference applies multiplicative noise `noise_std`.
- **amplitude_noiseaware**: Amplitude-coded output; training samples multiplicative noise from `train_noise_list`.

## Hyperparameters (config.yml)
- `digits_demo.*`: baseline MNIST sweep settings (epochs, lr, noise_std, methods).
- `fashion_complex.*`: Fashion-MNIST sweep settings.
- `kmnist_benchmark.*`, `emnist_letters_benchmark.*`, `cifar10_flat_benchmark.*`: dataset-specific settings (input_dim, num_classes, hidden_dims, epochs, lr, noise_std, methods).

## Scripts
- `src/run_benchmark.py`: trains all models with the baseline settings, sweeps inference noise, saves `results/benchmark_digits.csv/json`.
- `src/run_benchmark_fashion.py`: trains any variant selected via `--config-key` (Fashion-MNIST, KMNIST, EMNIST Letters, CIFAR-10, etc.), saves `results/benchmark_<config-key>.csv/json`.
- `src/benchmark_ui.py`: Gradio UI to run the benchmark with adjustable epochs, lr, noise list, and a dropdown to select the dataset/variant.
- `src/run_digits_demo.py`: smaller standalone sweep used in reproduce script.
