import os
import cv2
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# === Configuration ===
GT_LABELS_DIR = "metrics/tennis_ball_eval_dataset/labels"
PREDICTIONS_FILE = "metrics/predictions/pred_ball_coords.txt"
DIST_THRESHOLD = 0.02  # Euclidean distance threshold in normalized coordinates

# === Distance function between predicted and ground-truth center ===
def center_distance(pred, gt):
    px, py = pred
    gx, gy = gt
    return np.sqrt((px - gx) ** 2 + (py - gy) ** 2)

# === Load predictions ===
with open(PREDICTIONS_FILE, 'r') as f:
    pred_lines = f.read().strip().split('\n')

y_true = []
y_pred = []
predicted_images = set()

for line in pred_lines:
    parts = line.strip().split()
    if len(parts) < 2:
        continue

    filename = parts[0]
    predicted_images.add(filename)
    label_path = os.path.join(GT_LABELS_DIR, os.path.splitext(filename)[0] + ".txt")

    # Handle missing predictions (marked as NA NA NA NA)
    if parts[1] == "NA":
        if os.path.exists(label_path):
            # Ground truth exists but no prediction → False Negative
            y_true.append(1)
            y_pred.append(0)
        # If neither GT nor prediction exists, skip it (True Negative ignored)
        continue

    # Process valid prediction
    px, py = float(parts[1]), float(parts[2])
    pred_center = (px, py)
    matched = False

    if os.path.exists(label_path):
        with open(label_path, 'r') as gt_file:
            for line in gt_file:
                cls, x, y, w, h = map(float, line.strip().split())
                gt_center = (x, y)
                if center_distance(pred_center, gt_center) <= DIST_THRESHOLD:
                    matched = True
                    break

        y_true.append(1)
        y_pred.append(1 if matched else 0)
    else:
        # Prediction exists but ground truth does not → False Positive
        y_true.append(0)
        y_pred.append(1)

# === Calculate metrics ===
tp = sum((np.array(y_true) == 1) & (np.array(y_pred) == 1))
fp = sum((np.array(y_true) == 0) & (np.array(y_pred) == 1))
fn = sum((np.array(y_true) == 1) & (np.array(y_pred) == 0))

precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

# === Print summary ===
print("\nEvaluation: Center Distance Threshold")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1 Score:  {f1:.2f}")
print(f"True Positives (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")

import matplotlib.pyplot as plt

def plot_metrics(tp, fp, fn, precision, recall, f1, y_true, y_pred):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Bar chart: TP, FP, FN
    counts = [tp, fp, fn]
    axs[0, 0].bar(['TP', 'FP', 'FN'], counts, color=['green', 'red', 'orange'])
    axs[0, 0].set_title('Detection Counts')
    axs[0, 0].set_ylabel('Count')
    for i, count in enumerate(counts):
        axs[0, 0].text(i, count + 1, str(count), ha='center', va='bottom')

    # 2. Bar chart: precision, recall, F1
    metrics = [precision, recall, f1]
    axs[0, 1].bar(['Precision', 'Recall', 'F1'], metrics, color='steelblue')
    axs[0, 1].set_ylim(0, 1.05)
    axs[0, 1].set_ylabel('Score')
    axs[0, 1].set_title('Evaluation Metrics')
    for i, score in enumerate(metrics):
        axs[0, 1].text(i, score + 0.02, f"{score:.2f}", ha='center', va='bottom')

    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Ball", "Ball"])
    disp.plot(ax=axs[1, 0], colorbar=False)
    axs[1, 0].set_title("Confusion Matrix")

    # 4. Pie chart: distribution of predictions
    pie_labels = ['True Positives', 'False Positives', 'False Negatives']
    pie_counts = [tp, fp, fn]
    axs[1, 1].pie(pie_counts, labels=pie_labels, autopct='%1.1f%%',
                  colors=['green', 'red', 'orange'], startangle=90)
    axs[1, 1].set_title('Prediction Distribution')

    plt.suptitle("GridTrackNet Performance Overview", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

plot_metrics(tp, fp, fn, precision, recall, f1, y_true, y_pred)

