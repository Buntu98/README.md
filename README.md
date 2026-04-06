# README: Deep Generative models Assignment 1
## Overview

This project demonstrates autoregressive and normalizing flow models using PyTorch. It covers:

- MADE (Masked Autoencoder for Distribution Estimation) – for modeling binary MNIST and 2D synthetic datasets.
- MADE with Gaussian Mixture Conditionals – for modeling 2D synthetic moon datasets.
- RealNVP (Normalizing Flows) – for density estimation and sampling from 2D datasets.
- Visualization of densities and samples for qualitative evaluation.

The code is implemented to run end-to-end in Google Colab or any Python environment with PyTorch.

## Requirements
- Python 3.8+
- PyTorch
- torchvision
- numpy
- matplotlib
- scikit-learn

You can install dependencies with:

pip install torch torchvision numpy matplotlib scikit-learn

## Project Structure
- MNIST binary MADE
- Loads MNIST digits 0 and 1, resizes to 16×16, and binarizes.
- Trains a MADE model to learn the Bernoulli distribution of pixels.
- Includes compute_nll to evaluate test negative log-likelihood.
- Generates and visualizes 16 sample images from the trained model.
- 2D Moons Dataset with MADE
- Generates synthetic 2D moon-shaped data.
- Trains a simple autoregressive MADE and MADE with Gaussian Mixture conditionals.
- Visualizes density and samples.
- RealNVP (Normalizing Flows)
- Implements a RealNVP model with affine coupling layers.
- Trains on the synthetic 2D dataset using reverse KL divergence.
- Evaluates log-likelihood, plots density, and samples points from the learned distribution.
- Unnormalised Density and Reverse KL
- Defines an unnormalised 2D Gaussian mixture density.
- Trains RealNVP using reverse KL divergence between model and target distribution.
## Usage

Load MNIST and train MADE:

model = MADE(input_dim=256, hidden_dim=1024)

Generate samples:

samples = sample(model, n_samples=16)
plt.imshow(samples[0][0], cmap='gray')

Train on synthetic moons dataset:

X, y = make_moons(n_samples=1000, noise=0.1)
model = SimpleMADE(hidden=32)

Train RealNVP and visualize density:

model = RealNVP()

samples = model.sample(5000)

## Key Functions
MaskedLinear – Linear layer with an autoregressive mask.
MADE – Autoregressive model for high-dimensional binary data.
SimpleMADE / MADE_MoG – For 2D regression with Gaussian and Gaussian mixture conditionals.
RealNVP – Flow-based model for invertible transformations and density estimation.
compute_nll – Compute negative log-likelihood for model evaluation.
sample – Generate new samples from the trained model.
unnormalised_log_p – Defines target unnormalised 2D distribution for reverse KL training.

## Visualization

The project generates:

Samples from binary MNIST after MADE training.
Density plots and sample points for 2D synthetic datasets.
Contour plots of RealNVP-fitted densities.
Comparison between true data and generated samples.
