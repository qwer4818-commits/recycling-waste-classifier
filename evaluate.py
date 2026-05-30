"""
evaluate.py
-----------
Evaluate trained CNN model and visualize performance.

Tasks:
1. Measure overall test accuracy
2. Plot confusion matrix
3. Analyze per-class accuracy
4. Visualize wrong predictions (why did it fail?)
5. Print disposal guide using conditional statements

Usage:
    python3 source/evaluate.py

Requirements:
    models/best_model.pth must exist (run train.py first).
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

sys.path.append(os.path.dirname(__file__))
from train import RecyclingCNN

# ── Settings ──────────────────────────────────────────────────────────────────

CLASSES     = ["plastic", "paper", "glass", "metal", "styrofoam", "trash"]
NUM_CLASSES = len(CLASSES)

SPLIT_DIR   = os.path.join("data", "split")
MODEL_PATH  = os.path.join("models", "best_model.pth")
RESULTS_DIR = "results"

IMG_SIZE    = 224
BATCH_SIZE  = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Load Data ─────────────────────────────────────────────────────────────────

def load_test_data():
    """
    Load test dataset.

    Returns:
        tuple: (dataloader, dataset)
    """
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_dir   = os.path.join(SPLIT_DIR, "test")
    dataset    = datasets.ImageFolder(root=test_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)

    print(f"  Test dataset: {len(dataset)} images")
    print(f"  Classes: {dataset.classes}")
    return dataloader, dataset


# ── Load Model ────────────────────────────────────────────────────────────────

def load_model():
    """
    Load trained model weights.

    Returns:
        model: CNN model with loaded weights
    """
    if not os.path.exists(MODEL_PATH):
        print(f"  [ERROR] {MODEL_PATH} not found. Run train.py first.")
        exit(1)

    model = RecyclingCNN(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"  Model loaded: {MODEL_PATH}")
    return model


# ── Prediction ────────────────────────────────────────────────────────────────

def get_all_predictions(model, dataloader):
    """
    Get predictions for all test data.

    Args:
        model: trained CNN model
        dataloader: test data loader

    Returns:
        tuple: (true labels, predicted labels, probabilities)
    """
    all_labels = []
    all_preds  = []
    all_probs  = []

    with torch.no_grad():
        for images, labels in dataloader:
            images  = images.to(DEVICE)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, dim=1)

            all_labels.extend(labels.numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ── Disposal Guide ────────────────────────────────────────────────────────────

def print_disposal_guide(predicted_class):
    """
    Print disposal instructions based on predicted class.
    Uses if-elif-else conditional statements for decision rules.

    Args:
        predicted_class (str): predicted class name
    """
    print(f"\n  [{predicted_class}] Disposal Guide")
    print("  " + "-" * 35)

    if predicted_class == "plastic":
        print("  Plastic - Empty, rinse, and put in plastic recycling bin.")
    elif predicted_class == "paper":
        print("  Paper - Remove tape/stickers, put in paper recycling bin.")
    elif predicted_class == "glass":
        print("  Glass - Handle carefully, put in glass recycling bin.")
    elif predicted_class == "metal":
        print("  Metal - Empty, crush if possible, put in metal recycling bin.")
    elif predicted_class == "styrofoam":
        print("  Styrofoam - Remove foreign materials, put in styrofoam bin.")
    else:
        print("  Trash - Put in general waste bag.")


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(labels, preds, class_names):
    """
    Plot confusion matrix heatmap.

    How to read:
    - Diagonal: correct predictions (higher is better)
    - Off-diagonal: wrong predictions (lower is better)

    Args:
        labels (array): true labels
        preds (array): predicted labels
        class_names (list): class names
    """
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(8, 7))
    im      = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)

    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color=color, fontsize=12, fontweight="bold")

    ax.set_title("Confusion Matrix", fontsize=14, pad=15)
    ax.set_ylabel("True Class", fontsize=12)
    ax.set_xlabel("Predicted Class", fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_class_accuracy(labels, preds, class_names):
    """
    Plot per-class accuracy bar chart.

    Args:
        labels (array): true labels
        preds (array): predicted labels
        class_names (list): class names
    """
    class_acc = []

    for i in range(len(class_names)):
        mask = (labels == i)
        if mask.sum() == 0:
            class_acc.append(0.0)
            continue
        acc = (preds[mask] == labels[mask]).mean()
        class_acc.append(acc)

    colors = ["#4CAF50" if a >= 0.8 else "#FF9800" for a in class_acc]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars    = ax.bar(class_names, class_acc, color=colors,
                     edgecolor="white", linewidth=0.8)

    for bar, acc in zip(bars, class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01, f"{acc:.1%}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_title("Per-Class Accuracy", fontsize=14, pad=15)
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1.15)

    green_patch  = mpatches.Patch(color="#4CAF50", label="Above 80%")
    orange_patch = mpatches.Patch(color="#FF9800", label="Below 80%")
    ax.legend(handles=[green_patch, orange_patch,
                        plt.Line2D([0], [0], color="red",
                                   linestyle="--", label="80% baseline")])
    ax.axhline(y=0.8, color="red", linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "class_accuracy.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")

    return class_acc


def plot_wrong_predictions(model, dataset, class_names, num_samples=12):
    """
    Visualize misclassified samples for error analysis.

    Args:
        model: trained CNN model
        dataset: test dataset
        class_names (list): class names
        num_samples (int): number of wrong samples to display
    """
    wrong_images = []
    wrong_true   = []
    wrong_pred   = []

    for idx in range(len(dataset)):
        img_tensor, true_label = dataset[idx]
        img_input = img_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output     = model(img_input)
            pred_label = torch.argmax(output, dim=1).item()

        if pred_label != true_label:
            wrong_images.append(img_tensor)
            wrong_true.append(true_label)
            wrong_pred.append(pred_label)

        if len(wrong_images) >= num_samples:
            break

    if not wrong_images:
        print("  No wrong predictions!")
        return

    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return torch.clamp(tensor * std + mean, 0, 1)

    cols  = 4
    rows  = (len(wrong_images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.5))
    axes  = axes.flatten() if rows > 1 else axes.flatten() if cols > 1 else [axes]

    for i, (img, true, pred) in enumerate(zip(wrong_images, wrong_true, wrong_pred)):
        img_show = denormalize(img).permute(1, 2, 0).numpy()
        axes[i].imshow(img_show)
        axes[i].set_title(
            f"True: {class_names[true]}\nPred: {class_names[pred]}",
            color="red", fontsize=10)
        axes[i].axis("off")

    for i in range(len(wrong_images), len(axes)):
        axes[i].axis("off")

    plt.suptitle("Wrong Prediction Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "wrong_predictions.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    """Run the full evaluation pipeline."""

    print("=" * 55)
    print("  Recycling Waste CNN - Model Evaluation")
    print(f"  Device: {DEVICE}")
    print("=" * 55)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n  [Step 1] Loading data and model...")
    dataloader, dataset = load_test_data()
    model       = load_model()
    class_names = dataset.classes

    print("\n  [Step 2] Running predictions...")
    labels, preds, probs = get_all_predictions(model, dataloader)

    total_acc = (labels == preds).mean()
    print(f"\n  Test Accuracy: {total_acc:.4f} ({total_acc:.1%})")

    print("\n  Per-class report:")
    print(classification_report(labels, preds, target_names=class_names))

    print("\n  [Step 3] Saving visualizations...")
    plot_confusion_matrix(labels, preds, class_names)
    class_acc = plot_class_accuracy(labels, preds, class_names)
    plot_wrong_predictions(model, dataset, class_names)

    print("\n" + "=" * 55)
    print("  Final Results")
    print("=" * 55)
    print(f"  Overall Accuracy: {total_acc:.1%}")
    print()
    for cls, acc in zip(class_names, class_acc):
        status = "OK " if acc >= 0.8 else "Low"
        print(f"  [{status}]  {cls:<12}: {acc:.1%}")

    print("\n  Disposal Guide:")
    for cls in class_names:
        print_disposal_guide(cls)

    print("\n  Saved files:")
    print("  - results/confusion_matrix.png")
    print("  - results/class_accuracy.png")
    print("  - results/wrong_predictions.png")
    print("=" * 55)


if __name__ == "__main__":
    main()