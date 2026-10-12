# ship-detection
Ship detection using computer vision and machine learning techniques.
# Ship Detection Using SAR Imagery and U-Net

## Overview

This project implements a deep learning-based approach for ship detection in Synthetic Aperture Radar (SAR) imagery using a U-Net convolutional neural network. The model performs pixel-level image segmentation to identify ship regions in SAR images.

The project uses Python, PyTorch, Rasterio, NumPy, and Albumentations for image processing, model training, data augmentation, and performance evaluation.

## Objectives

- Process and normalize multi-band SAR imagery.
- Divide large images into smaller patches for model training.
- Train a U-Net model to segment ships from background regions.
- Apply data augmentation to improve model generalization.
- Generate predicted ship segmentation masks.
- Evaluate segmentation performance using Dice Coefficient and Intersection over Union (IoU).

## Technologies Used

- **Python** — Programming language
- **PyTorch** — Deep learning model development and training
- **U-Net** — Image segmentation architecture
- **Rasterio** — Geospatial raster image processing
- **NumPy** — Numerical computation and array manipulation
- **Albumentations** — Image and mask augmentation
- **Matplotlib** — Visualization of images and segmentation masks
- **Scikit-learn** — Training and validation dataset splitting

## Methodology

### 1. SAR Image Preprocessing

The notebook includes a normalization function that processes raster bands individually, scales valid pixel values, and handles NoData values.

### 2. Patch Generation

Large SAR images and their corresponding ground-truth masks are divided into patches of **256 × 256 pixels**, using a stride of 128 pixels.

This overlapping patch-based approach creates training samples from larger images.

### 3. Data Augmentation

The training dataset applies the following transformations:

- Horizontal flipping
- Vertical flipping
- Rotation up to 45 degrees

The same spatial transformations are applied to the image patches and their corresponding masks.

### 4. U-Net Model

The project implements a U-Net-style convolutional neural network with:

- Two input channels for SAR image data
- Convolutional blocks with batch normalization and ReLU activation
- Max-pooling for spatial downsampling
- Upsampling and skip connections for feature reconstruction
- A final convolutional layer for pixel-level prediction

### 5. Model Training

The model is trained using the Adam optimizer and a combined Binary Cross-Entropy and Dice loss function.

The notebook specifies:

- Training epochs: 50
- Batch size: 4
- Initial learning rate: 0.0001
- Validation split: 20%
- Learning-rate scheduling using `ReduceLROnPlateau`

The model with the lowest observed validation loss is saved during training.

### 6. Ship Segmentation and Visualization

The trained model processes overlapping image patches to generate a predicted segmentation mask for the full image. Overlapping predictions are averaged, and a threshold of 0.5 is used to produce a binary mask.

The notebook visualizes the original SAR image, ground-truth mask, and predicted ship mask.

### 7. Performance Evaluation

The notebook implements two segmentation metrics:

- **Dice Coefficient:** Measures overlap between the predicted and ground-truth ship masks.
- **Intersection over Union (IoU):** Measures the intersection relative to the union of the predicted and ground-truth regions.

Actual evaluation scores should be added after running the notebook successfully.

## Project Structure

```text
ship-detection/
├── README.md
└── ship-detection-final.ipynb
```

## How to Run

1. Clone or download this repository.
2. Install the required Python libraries.
3. Open the notebook in Jupyter Notebook, JupyterLab, or Kaggle.
4. Download and configure the SAR imagery and corresponding ground-truth mask datasets.
5. Update the dataset and output paths to match your environment.
6. Run the notebook cells in sequence to preprocess the data, train the model, generate predictions, and evaluate performance.

### Install Dependencies

```bash
pip install numpy matplotlib rasterio torch scikit-learn albumentations
```

A compatible PyTorch installation may require a platform-specific installation command, particularly when using a GPU.

## Dataset Requirements

The notebook references SAR imagery and a corresponding binary ship mask. The original dataset files are not included in this repository.

Configure the required raster files and image paths before running the notebook. Ensure that the SAR image and ground-truth mask have matching spatial dimensions and alignment.

## Current Limitations

- The notebook uses environment-specific Kaggle file paths that must be updated when running elsewhere.
- Training and evaluation depend on the availability and correct alignment of the input imagery and ground-truth masks.
- The repository does not currently include verified
