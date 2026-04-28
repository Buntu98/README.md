# Flow Matching for MNIST Inpainting

## Project Outline

This notebook demonstrates a Flow Matching approach for generative modeling, specifically applied to generate and inpaint MNIST digits 0 and 4. The project covers data preparation, U-Net model training, sample generation, and an analysis of inpainting performance with varying mask sizes.

### 1. Data Loading and Preprocessing
- Loading the MNIST dataset.
- Filtering for digits 0 and 4.
- Resizing images to 14x14 pixels and normalizing them to a [-1, 1] range.
- Splitting the filtered dataset into training, validation, and test sets.
- Visualizing a sample of the processed data.

### 2. U-Net Model Architecture and Training
- **Time Embedding**: A neural network module for encoding continuous time information into a high-dimensional vector.
- **U-Net**: A convolutional neural network architecture with skip connections, adapted to incorporate time embeddings for each layer, designed to predict velocity fields for flow matching.
- **Training Loop**: Implementation of a training procedure using the Flow Matching objective. The model learns to predict the velocity field between a noisy image and the target data, with training and validation losses monitored.

### 3. Sample Generation
- **Sampling Function**: A function that uses the trained U-Net model and an Euler ODE solver to generate new synthetic images from random noise.
- **Visualization**: Displaying a grid of generated samples and comparing them qualitatively with real images from the dataset.

### 4. Image Inpainting
- **Mask Creation**: Generation of a centered square mask to simulate missing regions in images.
- **Mask Application**: Demonstrating how the mask is applied to test images, creating corrupted inputs for the inpainting task.
- **Deterministic Inpainting**: An inpainting procedure based on flow matching, which iteratively refines the missing regions using the trained model.
- **Inpainting Visualization**: Showing original, masked, and multiple inpainted samples for selected digits.
- **Effect of Mask Size**: An analysis of how different mask sizes impact the quality of the inpainted images.

## Packages Used
- `torch`
- `torchvision`
- `matplotlib.pyplot`
- `torch.utils.data.Subset`
- `torch.utils.data.random_split`
- `torch.nn`
- `torch.nn.functional`
- `torch.utils.data.DataLoader`
- `math` a dataset format.
