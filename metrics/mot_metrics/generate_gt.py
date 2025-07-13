import os
import csv
import re
import sys

# Add path to the 'predict_utils' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

LABELS_DIR = "metrics/tennis_ball_eval_dataset/labels"
OUTPUT_CSV = "metrics/mot_metrics/tracking_eval/gt.csv"


# Regex para extraer el número de frame: video1_0005_jpg.rf.<hash>.txt → 5
frame_regex = re.compile(r"video\d+_(\d+)_jpg\.rf\..*\.txt")

with open(OUTPUT_CSV, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    for filename in sorted(os.listdir(LABELS_DIR)):
        match = frame_regex.match(filename)
        if not match:
            continue

        frame_id = int(match.group(1))
        label_path = os.path.join(LABELS_DIR, filename)

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, x, y, w, h = parts
                writer.writerow([frame_id, 0, float(x), float(y), float(w), float(h)])
