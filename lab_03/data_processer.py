import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

class MNISTDataset(Dataset):
    """Custom Dataset class for MNIST data"""

    def __init__(self, data, labels=None, transform=None):
        """
        Args:
            data: numpy array of pixel values (n_samples, 784)
            labels: numpy array of labels (n_samples,)
            transform: torchvision transforms to apply
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get image and reshape to 28x28
        image = self.data[idx].reshape(28, 28).astype(np.float32)

        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
        else:
            # Convert to tensor and add channel dimension
            image = torch.tensor(image).unsqueeze(0)  # Shape: (1, 28, 28)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return image, label
        else:
            return image

def load_mnist_data(data_dir='digit-recognizer'):
    """Load MNIST training and test data from CSV files"""

    # Load training data
    train_df = pd.read_csv(f'{data_dir}/train.csv')

    # Separate labels and pixel data
    train_labels = train_df['label'].values
    train_data = train_df.drop('label', axis=1).values

    # Load test data
    test_df = pd.read_csv(f'{data_dir}/test.csv')
    test_data = test_df.values

    print(f"Training data shape: {train_data.shape}")
    print(f"Training labels shape: {train_labels.shape}")
    print(f"Test data shape: {test_data.shape}")

    return train_data, train_labels, test_data

def create_data_loaders(train_data, train_labels, test_data, batch_size=64, validation_split=0.2):
    """Create PyTorch DataLoaders for training, validation, and test sets"""

    # Normalize pixel values from 0-255 to 0-1
    train_data = train_data / 255.0
    test_data = test_data / 255.0

    # Split training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        train_data, train_labels, test_size=validation_split, random_state=42, stratify=train_labels
    )

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {test_data.shape[0]} samples")

    # Define transforms for data augmentation
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
    ])

    # Create datasets
    train_dataset = MNISTDataset(X_train, y_train, transform=train_transform)
    val_dataset = MNISTDataset(X_val, y_val)
    test_dataset = MNISTDataset(test_data)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def get_sample_data(loader, n_samples=5):
    """Get a few samples from a data loader for visualization"""
    data_iter = iter(loader)
    images, labels = next(data_iter)

    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")

    # Return first n_samples
    return images[:n_samples], labels[:n_samples]