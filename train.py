"""
train.py
--------
CNN model training script for recycling waste classification.
Demonstrates programming concepts learned in EF1003:
conditionals, loops, functions, and modular structure.

CNN Architecture:
    Conv2d → BatchNorm → ReLU → MaxPool (3 blocks)
    → Flatten → Fully Connected → Dropout → Output

Usage:
    python3 source/train.py

Requirements:
    pip3 install torch torchvision matplotlib
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time

# ── Configuration ──────────────────────────────────────────────────────────

CLASSES     = ["plastic", "paper", "glass", "metal", "styrofoam", "trash"]
NUM_CLASSES = len(CLASSES)

SPLIT_DIR   = os.path.join("data", "split")
MODEL_DIR   = "models"
RESULTS_DIR = "results"

# Hyperparameters
BATCH_SIZE  = 32
NUM_EPOCHS  = 50
LEARNING_RATE = 0.001
IMG_SIZE    = 224

# Use GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Data Loading ───────────────────────────────────────────────────────────

def get_transforms():
    """
    Returns image transformation pipelines for train/val/test.

    torchvision.transforms principles:
    - Compose: Apply multiple transforms sequentially
    - ToTensor: Convert PIL image (0~255) → Tensor (0.0~1.0)
    - Normalize: (pixel - mean) / std → stabilize training
      Using ImageNet mean/std values

    Returns:
        dict: {"train": transform, "val": transform, "test": transform}
    """
    # Training: add augmentation to prevent overfitting
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),       # 50% chance flip
        transforms.RandomRotation(degrees=15),         # ±15° rotation
        transforms.ColorJitter(brightness=0.2,         # Random brightness
                               contrast=0.2),
        transforms.ToTensor(),                         # 0~255 → 0~1
        transforms.Normalize(                          # Standardize
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Validation/Test: no augmentation, pure evaluation
    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return {
        "train": train_transform,
        "val":   eval_transform,
        "test":  eval_transform,
    }


def load_datasets():
    """
    Load train/val/test datasets from data/split/ folder.

    torchvision.datasets.ImageFolder principle:
    - Automatically reads folder structure and creates class labels
    - Folder names are sorted alphabetically to assign class indices
      Example: glass=0, metal=1, paper=2, plastic=3, styrofoam=4, trash=5

    Returns:
        tuple: (dataloaders dict, dataset_sizes dict, class_names list)
    """
    transforms_dict = get_transforms()
    dataloaders = {}
    dataset_sizes = {}

    # Loop: load train/val/test
    for split in ["train", "val", "test"]:
        split_path = os.path.join(SPLIT_DIR, split)

        # Conditional: check if folder exists
        if not os.path.exists(split_path):
            print(f"  [ERROR] {split_path} folder not found. Run preprocess.py first.")
            exit(1)

        dataset = datasets.ImageFolder(
            root=split_path,
            transform=transforms_dict[split]
        )

        # Conditional: skip if dataset is empty
        if len(dataset) == 0:
            print(f"  [WARNING] {split} dataset is empty.")
            continue

        # shuffle: only True for training (randomize order each epoch)
        shuffle = (split == "train")

        dataloaders[split] = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=0      # 0 for Mac compatibility
        )
        dataset_sizes[split] = len(dataset)

    class_names = dataloaders["train"].dataset.classes
    print(f"  Class names: {class_names}")
    print(f"  Dataset sizes: {dataset_sizes}")

    return dataloaders, dataset_sizes, class_names


# ── CNN Model Definition ───────────────────────────────────────────────────

class RecyclingCNN(nn.Module):
    """
    CNN model for recycling waste classification.

    Architecture:
    [Input: 3×224×224]
        ↓
    [Conv Block 1]: Conv(3→32) → BatchNorm → ReLU → MaxPool(2×2)
        → Output: 32×112×112
        ↓
    [Conv Block 2]: Conv(32→64) → BatchNorm → ReLU → MaxPool(2×2)
        → Output: 64×56×56
        ↓
    [Conv Block 3]: Conv(64→128) → BatchNorm → ReLU → MaxPool(2×2)
        → Output: 128×28×28
        ↓
    [Flatten]: 128×28×28 = 100,352-dim vector
        ↓
    [FC1]: 100,352 → 512 → ReLU → Dropout(0.5)
        ↓
    [FC2]: 512 → 6 (num classes)
        ↓
    [Output: 6 class probabilities]

    Layer roles:
    - Conv2d: Extract features (edges, textures, patterns)
    - BatchNorm: Stabilize training (normalize batch outputs)
    - ReLU: Add non-linearity (negative → 0)
    - MaxPool: Reduce feature map size (keep important features)
    - Dropout: Prevent overfitting (randomly disable neurons)
    """

    def __init__(self, num_classes=6):
        super(RecyclingCNN, self).__init__()

        # Conv Block 1: Extract basic features (edges, colors)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=3, padding=1),  # 3×224×224 → 32×224×224
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # → 32×112×112
        )

        # Conv Block 2: Extract mid-level features (shapes, patterns)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=1),  # → 64×112×112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # → 64×56×56
        )

        # Conv Block 3: Extract high-level features (object types)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, padding=1),  # → 128×56×56
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # → 128×28×28
        )

        # Fully Connected Layers: Classify based on extracted features
        self.classifier = nn.Sequential(
            nn.Flatten(),                            # 128×28×28 → 100,352
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),                       # Prevent overfitting
            nn.Linear(512, num_classes),             # → 6 classes
        )

    def forward(self, x):
        """
        Forward pass function.
        Defines how input images pass through each layer.

        Args:
            x (Tensor): Input image batch [batch_size, 3, 224, 224]

        Returns:
            Tensor: Class scores [batch_size, num_classes]
        """
        x = self.conv_block1(x)   # Feature extraction stage 1
        x = self.conv_block2(x)   # Feature extraction stage 2
        x = self.conv_block3(x)   # Feature extraction stage 3
        x = self.classifier(x)    # Classification
        return x


# ── Training Functions ─────────────────────────────────────────────────────

def train_one_epoch(model, dataloader, criterion, optimizer):
    """
    Train the model for one epoch.

    Args:
        model: CNN model
        dataloader: Training data loader
        criterion: Loss function (CrossEntropyLoss)
        optimizer: Optimization algorithm (Adam)

    Returns:
        tuple: (average loss, accuracy)
    """
    model.train()  # Training mode (enable Dropout)
    running_loss = 0.0
    correct = 0
    total = 0

    # Loop: iterate through batches
    for images, labels in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # 1. Zero gradients
        optimizer.zero_grad()

        # 2. Forward pass
        outputs = model(images)

        # 3. Calculate loss
        loss = criterion(outputs, labels)

        # 4. Backward pass — compute gradients
        loss.backward()

        # 5. Update weights
        optimizer.step()

        # Record statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion):
    """
    Evaluate model performance (validation or test).

    Args:
        model: CNN model
        dataloader: Evaluation data loader
        criterion: Loss function

    Returns:
        tuple: (average loss, accuracy)
    """
    model.eval()   # Evaluation mode (disable Dropout)
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():   # Disable gradient computation (save memory)
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


# ── Visualization ──────────────────────────────────────────────────────────

def plot_training_history(history):
    """
    Save loss/accuracy graphs during training.

    Args:
        history (dict): Recorded train/val loss, accuracy per epoch
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss graph
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val Loss",   markersize=4)
    axes[0].set_title("Loss over Epochs", fontsize=13)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy graph
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Acc", markersize=4)
    axes[1].plot(epochs, history["val_acc"],   "r-o", label="Val Acc",   markersize=4)
    axes[1].set_title("Accuracy over Epochs", fontsize=13)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "training_history.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Training graph saved: {save_path}")


# ── Main Execution ─────────────────────────────────────────────────────────

def main():
    """Main function to run the entire training pipeline."""

    print("=" * 60)
    print("  Recycling Waste CNN Training")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load data
    print("\n  [Step 1] Loading datasets...")
    dataloaders, dataset_sizes, class_names = load_datasets()

    # 2. Initialize model
    print("\n  [Step 2] Initializing CNN model...")
    model = RecyclingCNN(num_classes=NUM_CLASSES).to(DEVICE)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 3. Setup loss function & optimizer
    criterion = nn.CrossEntropyLoss()        # Multi-class classification loss
    optimizer = optim.Adam(                  # Adam: adaptive learning rate
        model.parameters(),
        lr=LEARNING_RATE
    )
    # Learning rate scheduler: reduce lr if val loss doesn't improve for 3 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    # 4. Training loop
    print(f"\n  [Step 3] Training started ({NUM_EPOCHS} epochs)...")
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   []
    }

    best_val_acc = 0.0
    best_model_path = os.path.join(MODEL_DIR, "best_model.pth")

    start_time = time.time()

    # Loop: train for NUM_EPOCHS
    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, dataloaders["train"], criterion, optimizer
        )
        # Validate
        val_loss, val_acc = evaluate(
            model, dataloaders["val"], criterion
        )

        # Update learning rate scheduler
        scheduler.step(val_loss)

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Conditional: save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            saved_mark = "  <- SAVED"
        else:
            saved_mark = ""

        print(f"  Epoch [{epoch:2d}/{NUM_EPOCHS}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
              f"{saved_mark}")

    elapsed = time.time() - start_time
    print(f"\n  Training completed! Time: {elapsed/60:.1f} minutes")
    print(f"  Best Val Accuracy: {best_val_acc:.4f}")

    # 5. Save training graph
    plot_training_history(history)

    # 6. Save final model (last epoch)
    final_model_path = os.path.join(MODEL_DIR, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"  Final model saved: {final_model_path}")
    print(f"  Best model saved: {best_model_path}")

    print("\n" + "=" * 60)
    print("  Training complete! Next: python3 source/evaluate.py")
    print("=" * 60)


if __name__ == "__main__":
    main()