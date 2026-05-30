"""
preprocess.py
-------------
Preprocess raw images: quality filtering, deduplication, augmentation, and splitting.
Steps: Remove low-quality/duplicates → Balance classes → Resize/Normalize → Augment → Train/Val/Test split

Usage:
    python3 source/preprocess.py

Requirements:
    pip3 install Pillow imagehash matplotlib scikit-learn tqdm numpy
"""

import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import imagehash
from PIL import Image, ImageEnhance, ImageOps
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ── Configuration ──────────────────────────────────────────────────────────

CLASSES = ["plastic", "paper", "glass", "metal", "styrofoam", "trash"]

RAW_DIR       = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
SPLIT_DIR     = os.path.join("data", "split")

IMG_SIZE       = (224, 224)
MIN_SIZE       = 100
MAX_FILE_KB    = 5
HASH_THRESHOLD = 10

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

TARGET_PER_CLASS = 300

random.seed(42)
np.random.seed(42)


# ── Step 1: Filter low-quality and duplicate images ────────────────────────

def is_low_quality(filepath):
    """
    Check if image is low quality.

    Args:
        filepath (str): Image file path

    Returns:
        bool: True if low quality
    """
    file_size_kb = os.path.getsize(filepath) / 1024
    if file_size_kb < MAX_FILE_KB:
        return True

    try:
        img = Image.open(filepath)
        width, height = img.size
        if width < MIN_SIZE or height < MIN_SIZE:
            return True
        if img.mode not in ("RGB", "RGBA", "L"):
            return True
    except Exception:
        return True

    return False


def remove_duplicates(file_list):
    """
    Remove duplicate images using perceptual hashing.
    imagehash.average_hash: Resize to 8x8, compare with average brightness.

    Args:
        file_list (list): List of image file paths

    Returns:
        list: List with duplicates removed
    """
    seen_hashes = []
    unique_files = []

    for filepath in tqdm(file_list, desc="    Checking duplicates", leave=False):
        try:
            img = Image.open(filepath)
            h = imagehash.average_hash(img)

            is_duplicate = False
            for seen_hash in seen_hashes:
                if abs(h - seen_hash) <= HASH_THRESHOLD:
                    is_duplicate = True
                    break

            if not is_duplicate:
                seen_hashes.append(h)
                unique_files.append(filepath)

        except Exception:
            continue

    return unique_files


def filter_class(class_name):
    """
    Filter low-quality and duplicate images for a specific class.

    Args:
        class_name (str): Class name

    Returns:
        list: List of filtered file paths
    """
    raw_path = os.path.join(RAW_DIR, class_name)

    if not os.path.exists(raw_path):
        print(f"  [{class_name}] Folder not found — skipping")
        return []

    all_files = [
        os.path.join(raw_path, f)
        for f in os.listdir(raw_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not all_files:
        print(f"  [{class_name}] No images — skipping (take photos and run again)")
        return []

    original_count = len(all_files)
    quality_passed = [f for f in all_files if not is_low_quality(f)]
    removed_quality = original_count - len(quality_passed)
    unique_files = remove_duplicates(quality_passed)
    removed_duplicate = len(quality_passed) - len(unique_files)

    print(f"  [{class_name}] Original {original_count} "
          f"-> Removed low-quality -{removed_quality} "
          f"-> Removed duplicates -{removed_duplicate} "
          f"-> Final {len(unique_files)}")

    return unique_files


# ── Step 2: Augmentation ────────────────────────────────────────────────────

def augment_image(img):
    """
    Apply random augmentation to image.
    Waste photos can be taken from any angle, so augmentation is natural.

    Args:
        img (PIL.Image): Original image

    Returns:
        PIL.Image: Augmented image
    """
    if random.random() > 0.5:
        img = ImageOps.mirror(img)
    if random.random() > 0.7:
        img = ImageOps.flip(img)

    angle = random.uniform(-30, 30)
    img = img.rotate(angle, expand=False, fillcolor=(128, 128, 128))

    brightness_factor = random.uniform(0.7, 1.3)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)

    contrast_factor = random.uniform(0.8, 1.2)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    return img


def resize_and_normalize(img):
    """
    Resize image to 224x224 and convert to RGB.

    Args:
        img (PIL.Image): Original image

    Returns:
        PIL.Image: Preprocessed image
    """
    if img.mode == "RGBA":
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    img = img.resize(IMG_SIZE, Image.LANCZOS)
    return img


def process_and_augment_class(class_name, valid_files):
    """
    Apply preprocessing and augmentation, then save images.

    Args:
        class_name (str): Class name
        valid_files (list): List of filtered file paths
    """
    # Skip if no valid files
    if not valid_files:
        print(f"  [{class_name}] No images — skipping (take photos and run again)")
        return 0

    save_dir = os.path.join(PROCESSED_DIR, class_name)
    os.makedirs(save_dir, exist_ok=True)

    saved = 0

    # Preprocess and save original images
    for filepath in tqdm(valid_files, desc=f"    [{class_name}] Processing", leave=False):
        try:
            img = Image.open(filepath)
            img = resize_and_normalize(img)
            save_path = os.path.join(save_dir, f"{class_name}_{saved:04d}.jpg")
            img.save(save_path, "JPEG", quality=90)
            saved += 1
        except Exception:
            continue

    # If below target, augment to fill
    aug_index = saved
    attempts = 0
    max_attempts = TARGET_PER_CLASS * 3  # Prevent infinite loop

    while saved < TARGET_PER_CLASS and attempts < max_attempts:
        source_path = random.choice(valid_files)
        try:
            img = Image.open(source_path)
            img = resize_and_normalize(img)
            img = augment_image(img)

            save_path = os.path.join(save_dir, f"{class_name}_aug_{aug_index:04d}.jpg")
            img.save(save_path, "JPEG", quality=90)
            saved += 1
            aug_index += 1
        except Exception:
            pass
        attempts += 1

    print(f"  [{class_name}] Saved to processed/: {saved} images")
    return saved


# ── Step 3: Visualize class distribution ────────────────────────────────────

def plot_class_distribution(stage="processed"):
    """
    Visualize number of images per class as bar chart.

    Args:
        stage (str): "raw" or "processed"
    """
    base = RAW_DIR if stage == "raw" else PROCESSED_DIR
    counts = {}

    for cls in CLASSES:
        folder = os.path.join(base, cls)
        if os.path.exists(folder):
            count = len([f for f in os.listdir(folder)
                         if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            counts[cls] = count
        else:
            counts[cls] = 0

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336", "#795548"]
    bars = ax.bar(counts.keys(), counts.values(), color=colors, edgecolor="white", linewidth=0.8)

    for bar, count in zip(bars, counts.values()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 3,
                str(count),
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_title(f"Images per Class ({stage} stage)", fontsize=14, pad=15)
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.axhline(y=TARGET_PER_CLASS, color="red", linestyle="--",
               alpha=0.7, label=f"Target: {TARGET_PER_CLASS}")
    ax.legend()
    plt.tight_layout()

    save_path = os.path.join("results", f"class_distribution_{stage}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Distribution graph saved: {save_path}")


# ── Step 4: Split into Train/Val/Test ────────────────────────────────────────

def split_dataset():
    """
    Split processed/ images into train/val/test.
    Uses stratified split to maintain class proportions in each set.
    """
    print("\n  [Step 4] Splitting into train/val/test...")

    split_counts = {"train": 0, "val": 0, "test": 0}

    for cls in CLASSES:
        src_dir = os.path.join(PROCESSED_DIR, cls)
        if not os.path.exists(src_dir):
            print(f"  [Skipped] {cls}: processed folder not found")
            continue

        all_files = [
            f for f in os.listdir(src_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(all_files) < 10:
            print(f"  [Skipped] {cls}: insufficient images ({len(all_files)})")
            continue

        train_files, valtest_files = train_test_split(
            all_files, test_size=(VAL_RATIO + TEST_RATIO), random_state=42
        )

        val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
        val_files, test_files = train_test_split(
            valtest_files, test_size=(1 - val_ratio_adjusted), random_state=42
        )

        for split_name, file_list in [("train", train_files),
                                       ("val",   val_files),
                                       ("test",  test_files)]:
            dest_dir = os.path.join(SPLIT_DIR, split_name, cls)
            os.makedirs(dest_dir, exist_ok=True)
            for fname in file_list:
                shutil.copy2(os.path.join(src_dir, fname),
                             os.path.join(dest_dir, fname))
            split_counts[split_name] += len(file_list)

        print(f"  [{cls}] train:{len(train_files)} / val:{len(val_files)} / test:{len(test_files)}")

    print(f"\n  Split complete -> train:{split_counts['train']} / "
          f"val:{split_counts['val']} / test:{split_counts['test']}")


# ── Main Execution ──────────────────────────────────────────────────────────

def main():
    """Main function to run the entire preprocessing pipeline."""

    print("=" * 60)
    print("  Recycling Waste Image Preprocessing Pipeline")
    print("=" * 60)

    os.makedirs("results", exist_ok=True)

    # Step 1: Visualize raw distribution
    print("\n  [Step 1] Checking raw data distribution...")
    plot_class_distribution(stage="raw")

    # Step 2: Filtering
    print("\n  [Step 2] Filtering low-quality and duplicate images...")
    valid_files_per_class = {}
    for cls in CLASSES:
        valid_files_per_class[cls] = filter_class(cls)

    # Step 3: Preprocessing + Augmentation
    print("\n  [Step 3] Applying resize and augmentation...")
    for cls in CLASSES:
        process_and_augment_class(cls, valid_files_per_class[cls])

    plot_class_distribution(stage="processed")

    # Step 4: Split
    split_dataset()

    print("\n" + "=" * 60)
    print("  Preprocessing complete! Next: python3 source/train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()