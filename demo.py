"""
demo.py
-------
Real-time recycling waste classifier using webcam and trained CNN model.
Displays class probabilities and disposal guide in real-time.

Usage:
    python3 source/demo.py

Requirements:
    pip3 install opencv-python torch torchvision
    models/best_model.pth must exist.

Controls:
    Q key -> Quit
"""

import os
import sys
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

# Import model class from train.py
sys.path.append(os.path.dirname(__file__))
from train import RecyclingCNN

# ── Settings ──────────────────────────────────────────────────────────────────

CLASSES = ["plastic", "paper", "glass", "metal", "styrofoam", "trash"]

# Disposal guide messages
DISPOSAL_GUIDE = {
    "plastic":   "Empty, rinse, and put in plastic bin",
    "paper":     "Remove tape/stickers, put in paper bin",
    "glass":     "Handle carefully, put in glass bin",
    "metal":     "Empty, crush, and put in metal bin",
    "styrofoam": "Remove foreign materials, put in styrofoam bin",
    "trash":     "Put in general waste bag",
}

# Class colors (BGR format)
COLORS = {
    "plastic":   (255, 100,  50),
    "paper":     ( 50, 200,  50),
    "glass":     (200, 200,  50),
    "metal":     (100, 100, 255),
    "styrofoam": (200,  50, 200),
    "trash":     (150, 150, 150),
}

MODEL_PATH = os.path.join("models", "best_model.pth")
IMG_SIZE   = 224
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Load Model ────────────────────────────────────────────────────────────────

def load_model():
    """
    Load the trained CNN model from saved weights.

    Returns:
        model: CNN model with loaded weights
    """
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] {MODEL_PATH} not found. Run train.py first.")
        exit(1)

    model = RecyclingCNN(num_classes=len(CLASSES)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Model loaded: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    return model


# ── Preprocessing ─────────────────────────────────────────────────────────────

# Transform pipeline: camera frame -> CNN input tensor
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def preprocess_frame(frame):
    """
    Convert camera frame to model input tensor.

    Args:
        frame: OpenCV BGR frame

    Returns:
        Tensor: model input [1, 3, 224, 224]
    """
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    tensor  = transform(pil_img).unsqueeze(0).to(DEVICE)
    return tensor


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(model, frame):
    """
    Run prediction on a single frame.

    Args:
        model: trained CNN model
        frame: OpenCV camera frame

    Returns:
        tuple: (predicted class name, probability array)
    """
    tensor = preprocess_frame(frame)

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    pred_idx   = np.argmax(probs)
    pred_class = CLASSES[pred_idx]

    return pred_class, probs


# ── Draw UI ───────────────────────────────────────────────────────────────────

def draw_ui(frame, pred_class, probs):
    """
    Draw prediction results and probability bars on the frame.

    Args:
        frame: OpenCV camera frame
        pred_class (str): predicted class name
        probs (array): probability values for each class

    Returns:
        frame: frame with UI drawn
    """
    h, w   = frame.shape[:2]
    color  = COLORS[pred_class]
    conf   = probs[CLASSES.index(pred_class)]

    # ── Top info bar
    cv2.rectangle(frame, (0, 0), (w, 80), (30, 30, 30), -1)
    cv2.putText(frame,
                f"Class: {pred_class.upper()}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(frame,
                f"Confidence: {conf:.1%}",
                (15, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)

    # ── Right panel: probability bars
    panel_x = w - 260
    cv2.rectangle(frame, (panel_x - 10, 80), (w, h), (20, 20, 20), -1)

    # Loop through each class and draw probability bar
    for i, (cls, prob) in enumerate(zip(CLASSES, probs)):
        y         = 105 + i * 55
        bar_color = COLORS[cls]

        cv2.putText(frame, cls,
                    (panel_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

        # Bar background
        cv2.rectangle(frame,
                      (panel_x, y + 8),
                      (panel_x + 200, y + 30),
                      (60, 60, 60), -1)

        # Bar fill
        bar_w = int(200 * prob)
        if bar_w > 0:
            cv2.rectangle(frame,
                          (panel_x, y + 8),
                          (panel_x + bar_w, y + 30),
                          bar_color, -1)

        # Probability text
        cv2.putText(frame,
                    f"{prob:.1%}",
                    (panel_x + 205, y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # ── Bottom disposal guide (bigger text)
    guide = DISPOSAL_GUIDE[pred_class]
    cv2.rectangle(frame, (0, h - 75), (panel_x - 15, h), (30, 30, 30), -1)
    cv2.putText(frame,
                "How to dispose:",
                (10, h - 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
    cv2.putText(frame,
                guide,
                (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # ── Border color based on predicted class
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 3)

    return frame


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    """Main function to run the real-time demo."""

    print("=" * 50)
    print("  Recycling Waste Classifier - Live Demo")
    print("=" * 50)
    print("  Press Q to quit.")

    model = load_model()

    # Open webcam (0 = default MacBook camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        exit(1)

    print("\n  Camera started! Show waste items to the camera.")

    # Main loop: process frame by frame
    while True:
        ret, frame = cap.read()

        if not ret:
            print("[WARNING] Cannot read frame.")
            break

        # Run prediction
        pred_class, probs = predict(model, frame)

        # Draw UI on frame
        frame = draw_ui(frame, pred_class, probs)

        # Show frame
        cv2.imshow("Recycling Waste Classifier - Press Q to quit", frame)

        # Quit on Q key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n  Demo ended.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()