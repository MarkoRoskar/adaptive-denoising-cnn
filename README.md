# EPSFilterNet: learned adaptive image denoising

A small PyTorch network that removes noise from RGB images by learning a bank of
convolution filters plus a per-pixel map that decides which filters to apply where.
Built as a university image-processing assignment.

## Results

Evaluated on 1000 freshly generated test images per noise type (noise std drawn uniformly
from 0.1–0.3 of the intensity range):

| Noise type     | PSNR noisy        | PSNR denoised     | SNR noisy         | SNR denoised      |
|----------------|-------------------|-------------------|-------------------|-------------------|
| Additive       | 15.60 ± 2.45 dB   | **27.38 ± 2.16 dB** | 10.67 ± 2.67 dB | **22.45 ± 2.40 dB** |
| Multiplicative | 20.21 ± 2.61 dB   | **30.42 ± 2.54 dB** | 15.30 ± 2.50 dB | **25.51 ± 2.27 dB** |

That is roughly a **+10–12 dB** improvement with a model of only **~50k parameters**.

## How it works

The network has two branches:

```
              ┌── Filter branch ──────────┐
noisy RGB ──► │ 8 learned 11×11 filters   │──► 8 filtered versions per channel ─┐
  image       │ per colour channel (R,G,B)│                                     │
              └───────────────────────────┘                                     ▼
              ┌── Classification branch ──┐                        per-pixel weighted sum
              │ 3 conv blocks (conv-BN-   │──► softmax over 8 ──►  of filter outputs  ──► denoised
              │ dropout-ReLU) + 1×1 conv  │    "class maps"                               image
              └───────────────────────────┘
```

- **Filter branch:** each colour channel is convolved with its own set of 8 learned
  11×11 kernels.
- **Classification branch:** a small CNN looks at the noisy image and outputs, for every
  pixel, a softmax distribution over the 8 filters.
- **Output:** each pixel is the weighted mix of the filter responses. Flat regions can
  get strong smoothing, while edges and thin lines get filters that keep them sharp.
  This makes it a spatially adaptive filter rather than a single global one.

Trained with MSE loss for 30 epochs, with a separate model for each noise type.

## Data

All training and test data is synthetic, generated on the fly:

- **Images** (`generate_image.py`): 256×256 canvases with a random background colour and
  randomly placed, overlapping filled triangles, quadrilaterals, ellipses and thin line
  "stars". This gives a mix of flat regions, sharp edges and fine detail.
- **Noise** (`generate_noise.py`):
  - *additive*: `I + N(0, σ)`
  - *multiplicative*: `I · (1 + N(0, σ))`, so the noise scales with brightness

## Project structure

| File                  | Purpose                                                        |
|-----------------------|----------------------------------------------------------------|
| `neural_network.py`   | Model (`EPSFilterNet`), training loop, SNR/PSNR evaluation     |
| `generate_image.py`   | Synthetic test image generator                                 |
| `generate_noise.py`   | Additive and multiplicative Gaussian noise                     |
| `generate_dataset.py` | Builds (noisy, clean) pairs and saves them as `.pt` files      |
| `report.html`         | Exported Jupyter notebook with the full run, figures and results (in Slovenian) |

## Usage

Requirements: `torch`, `numpy`, `opencv-python`, `matplotlib`.

```python
import torch
from generate_dataset import generate_dataset, NoiseType
from generate_noise import noise_add
from neural_network import EPSFilterNet, nn_train, nn_eval

device = "cuda" if torch.cuda.is_available() else "cpu"

generate_dataset(dataset_size=1000, noise_type=NoiseType.ADDITIVE)  # -> train_dataset_add.pt

model = EPSFilterNet(n_filts=8)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
nn_train(model, optimizer, torch.nn.MSELoss(), epochs=30,
         noise_fn=noise_add, batch_size=16, device=device)

model.eval()
nn_eval(model, iterations=100, noise_type=NoiseType.ADDITIVE, device=device)
```

The dataset size, learning rate and batch size above are example values.
