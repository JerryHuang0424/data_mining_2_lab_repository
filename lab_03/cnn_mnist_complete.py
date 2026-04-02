# Complete CNN Implementation for MNIST Digit Recognition
# This file contains the complete implementation that can be run as a Python script

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data processing functions
class MNISTDataset(Dataset):
    """Custom Dataset class for MNIST data"""

    def __init__(self, data, labels=None, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx].reshape(28, 28).astype(np.float32)

        if self.transform:
            image = self.transform(image)
        else:
            # Convert to tensor and add channel dimension if no transform
            image = torch.tensor(image).unsqueeze(0)  # Shape: (1, 28, 28)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return image, label
        else:
            return image

def load_mnist_data(data_dir='digit-recognizer'):
    """Load MNIST training and test data from CSV files"""
    train_df = pd.read_csv(f'{data_dir}/train.csv')
    train_labels = train_df['label'].values
    train_data = train_df.drop('label', axis=1).values

    test_df = pd.read_csv(f'{data_dir}/test.csv')
    test_data = test_df.values

    print(f"Training data shape: {train_data.shape}")
    print(f"Training labels shape: {train_labels.shape}")
    print(f"Test data shape: {test_data.shape}")

    return train_data, train_labels, test_data

def create_data_loaders(train_data, train_labels, test_data, batch_size=64, validation_split=0.2):
    """Create PyTorch DataLoaders for training, validation, and test sets"""
    train_data = train_data / 255.0
    test_data = test_data / 255.0

    X_train, X_val, y_train, y_val = train_test_split(
        train_data, train_labels, test_size=validation_split, random_state=42, stratify=train_labels
    )

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {test_data.shape[0]} samples")

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
    ])

    train_dataset = MNISTDataset(X_train, y_train, transform=train_transform)
    val_dataset = MNISTDataset(X_val, y_val)
    test_dataset = MNISTDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# CNN Model
class CNN_MNIST(nn.Module):
    def __init__(self):
        super(CNN_MNIST, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.drop1 = nn.Dropout(0.25)

        # Second convolutional block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.drop2 = nn.Dropout(0.25)

        # Third convolutional block
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, padding=1)
        self.drop3 = nn.Dropout(0.25)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.drop4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool3(x)
        x = self.drop3(x)

        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.drop4(x)
        x = self.fc2(x)

        return x

# Training functions
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total

    return train_loss, train_acc

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100. * correct / total

    return val_loss, val_acc

def train_model(model, train_loader, val_loader, epochs=10, learning_rate=0.001):
    """Train the model and track progress"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    best_val_acc = 0.0
    best_model_state = None

    print("Starting training...")
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        print('-' * 50)

    model.load_state_dict(best_model_state)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

    return train_losses, train_accs, val_losses, val_accs

def generate_predictions(model, test_loader, device):
    """Generate predictions for test set"""
    model.eval()
    predictions = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            predictions.extend(predicted.cpu().numpy())

    return predictions

def create_submission(predictions, output_file='submission.csv'):
    """Create competition submission file"""
    submission = pd.DataFrame({
        'ImageId': range(1, len(predictions) + 1),
        'Label': predictions
    })

    submission.to_csv(output_file, index=False)
    print(f"Submission file saved as {output_file}")
    return submission

# Main execution
if __name__ == "__main__":
    # Load data
    print("Loading MNIST data...")
    train_data, train_labels, test_data = load_mnist_data()

    # Create data loaders
    batch_size = 64
    train_loader, val_loader, test_loader = create_data_loaders(train_data, train_labels, test_data, batch_size=batch_size)

    # Create model
    model = CNN_MNIST().to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Train model
    print("Training the CNN model...")
    train_losses, train_accs, val_losses, val_accs = train_model(
        model, train_loader, val_loader, epochs=15, learning_rate=0.001
    )

    # Save trained model
    torch.save(model.state_dict(), 'mnist_cnn_model.pth')
    torch.save(model, 'mnist_cnn_full.pth')
    print("Model saved as 'mnist_cnn_model.pth' and 'mnist_cnn_full.pth'")

    # Save training metrics for later visualization
    np.savez('training_metrics.npz',
             train_losses=train_losses,
             train_accs=train_accs,
             val_losses=val_losses,
             val_accs=val_accs)
    print("Training metrics saved as 'training_metrics.npz'")

    # Generate predictions
    print("Generating test predictions...")
    predictions = generate_predictions(model, test_loader, device)

    # Create submission
    submission = create_submission(predictions)

    # Visualize results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

    print("\nCNN implementation completed successfully!")
    print(f"Final predictions generated for {len(predictions)} test images")
    print("Check 'submission.csv' for competition submission")