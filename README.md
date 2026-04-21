# Score-Based Generative Models (Diffusion Models)

This notebook implements a score-based generative model, also known as a diffusion model, for both unconditional and conditional data generation. The core idea is to learn the score function (gradient of the log-probability density) of data perturbed by noise at various levels, and then use this learned score function to reverse the diffusion process to generate new data samples.

## Project Description

This project explores the training and sampling mechanisms of score-based generative models. It demonstrates:

1.  **Unconditional Score Model (VE-SDE)**: Training a `ScoreNet` to model the score function of a noisy data distribution (specifically, the pinwheel dataset) and generating new samples using an ancestral sampler.
2.  **Conditional Score Model (VE-SDE) with Classifier-Free Guidance (CFG)**: Extending the unconditional model to a `CondScoreNet` that can generate samples conditioned on a class label. This model is trained with label dropout and utilizes Classifier-Free Guidance during sampling to improve sample quality and alignment with conditioning.

## Packages to Install

To run this notebook, the following Python package needs to be installed:

*   `torchdiffeq`: Used for solving differential equations, which can be part of advanced SDE sampling methods.

You can install it using pip:

```bash
!pip install torchdiffeq
```

## Key Imports

The following libraries are extensively used throughout the notebook:

*   `sys`: Standard system-specific parameters and functions.
*   `numpy as np`: For numerical operations, especially data generation.
*   `torch`: The primary deep learning framework.
*   `torch.nn as nn`: For building neural network architectures.
*   `torch.nn.functional as F`: For common neural network operations.
*   `matplotlib.pyplot as plt`: For plotting and visualization.
*   `torch.utils.data.DataLoader`: For efficient batching of data during training.
*   `torch.utils.data.TensorDataset`: For wrapping tensors into a dataset format.
