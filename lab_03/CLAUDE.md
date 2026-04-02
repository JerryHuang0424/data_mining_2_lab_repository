# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is Lab 03 of a Data Mining 2 course focused on digit handwriting identification using the MNIST dataset. The project involves building and evaluating machine learning models to classify handwritten digits (0-9) from 28x28 pixel grayscale images.

## Environment Setup

**Always use the conda pytorch environment** when working with this project. The environment contains all necessary dependencies for PyTorch-based machine learning workflows.

To activate the environment:
```bash
conda activate pytorch
```

## Project Structure

- `digit_handwriting_identify.ipynb` - Main Jupyter notebook containing the complete digit recognition workflow
- `data_processer.py` - Data processing module for loading, preprocessing, and augmenting MNIST data
- `digit-recognizer/` - Directory containing competition data files (`train.csv`, `test.csv`)
- `submission.csv` - Generated submission file with test predictions (created after running the notebook)
- `mnist_cnn_model.pth` - Saved model weights (PyTorch state_dict)
- `mnist_cnn_full.pth` - Full saved model (architecture + weights)

## Key Components

### Data Description
- **Training data**: `train.csv` with 785 columns (1 label column + 784 pixel columns), 42,000 samples
- **Test data**: `test.csv` with 784 pixel columns (no labels), 28,000 samples
- Each image is 28x28 pixels (784 total pixels)
- Pixel values range from 0-255 (grayscale)
- Label column contains digits 0-9

### Data Processing Pipeline (`data_processer.py`)
The data processing module provides the following functionality:

1. **Data Loading**: Loads CSV files and separates labels from pixel data
2. **Normalization**: Scales pixel values from 0-255 to 0-1 range
3. **Train/Validation Split**: Stratified split (80/20) maintaining class distribution
4. **Data Augmentation**: Random rotation (±10°) and translation (±10%) for training data
5. **Dataset Creation**: PyTorch Dataset class that reshapes flattened vectors to 28x28 images
6. **DataLoader Creation**: Creates batched data loaders for training, validation, and test sets

### CNN Model Architecture
The notebook implements a convolutional neural network with the following structure:

```
Input (1×28×28)
   ↓
Conv Block 1:
  • Conv2d(1→32, kernel=3, padding=1) + BatchNorm + ReLU
  • Conv2d(32→32, kernel=3, padding=1) + BatchNorm + ReLU
  • MaxPool2d(2) + Dropout(0.25)
   ↓
Conv Block 2:
  • Conv2d(32→64, kernel=3, padding=1) + BatchNorm + ReLU
  • Conv2d(64→64, kernel=3, padding=1) + BatchNorm + ReLU
  • MaxPool2d(2) + Dropout(0.25)
   ↓
Conv Block 3:
  • Conv2d(64→128, kernel=3, padding=1) + BatchNorm + ReLU
  • MaxPool2d(2, padding=1) + Dropout(0.25)
   ↓
Flatten → 128×4×4 = 2048 features
   ↓
Fully Connected Layers:
  • Linear(2048→256) + ReLU + Dropout(0.5)
  • Linear(256→10) → Output (10 classes)
```

**Total Parameters**: ~666,602

### Training Process
1. **Loss Function**: Cross-entropy loss
2. **Optimizer**: Adam with weight decay (1e-5)
3. **Learning Rate Scheduling**: StepLR (reduce by 0.5 every 5 epochs)
4. **Batch Size**: 64
5. **Epochs**: 15 (with early stopping based on validation accuracy)
6. **Validation**: 20% of training data held out for validation
7. **Best Model Selection**: Model with highest validation accuracy saved
8. **Model Persistence**: Trained model weights saved to disk (`mnist_cnn_model.pth` and `mnist_cnn_full.pth`)

### Evaluation Metrics
- **Primary Metric**: Categorization accuracy (percentage of correctly classified images)
- **Validation Metrics**: Loss and accuracy curves, confusion matrix, classification report
- **Test Output**: Submission file with ImageId-Label pairs for all 28,000 test images

## Common Development Tasks

### Running the Complete Pipeline
```bash
jupyter notebook digit_handwriting_identify.ipynb
```
Execute all cells in order to:
1. Load and visualize data
2. Train the CNN model
3. Evaluate performance
4. Generate test predictions

### Data Processing
- The `data_processer` module handles all data preparation
- Modify `create_data_loaders()` to adjust batch size, validation split, or augmentation
- Use `get_sample_data()` for quick visualization of data loader contents

### Model Development
- Modify `CNN_MNIST` class in the notebook to experiment with different architectures
- Adjust hyperparameters in `train_model()` function (epochs, learning rate, etc.)
- Add new training techniques (different optimizers, schedulers, regularization)

### Evaluation and Analysis
- The notebook includes comprehensive visualizations:
  - Sample images and label distribution
  - Training/validation loss and accuracy curves
  - Confusion matrix and classification report
  - Correct vs. incorrect prediction examples
- Use these to diagnose model performance and identify improvement areas

## Important Notes

- This is a Kaggle-style competition project; the goal is to achieve high test accuracy
- The notebook is designed for educational purposes—each step is documented and explained
- The data processing module demonstrates proper ML pipeline construction
- Focus on both implementation and understanding of why certain methods work well
- The CNN architecture follows modern best practices (batch norm, dropout, increasing filters)
- Random seeds are set for reproducibility (42)

## Expected Results

When running the complete notebook:
- Validation accuracy should reach ~99% after 15 epochs
- Training time: ~2-3 minutes per epoch on GPU, ~5-7 minutes per epoch on CPU
- Submission file `submission.csv` will be created in the project root
- All visualizations will be displayed inline in the notebook
- Model files will be saved for future use (`mnist_cnn_model.pth`, `mnist_cnn_full.pth`)

## Troubleshooting

- **CUDA out of memory**: Reduce batch size in `create_data_loaders()`
- **Slow training**: Ensure you're using GPU (`device` will show "cuda" if available)
- **Poor accuracy**: Check data normalization, model architecture, or training hyperparameters
- **File not found errors**: Verify `digit-recognizer/` directory contains `train.csv` and `test.csv`