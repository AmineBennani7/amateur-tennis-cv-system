import os
import csv
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from trackers.predict_utils import getPredictions

INPUT_FILE = "metrics/predictions/pred_ball_coords.txt"
OUTPUT_CSV = "metrics/mot_metrics/ball_metrics/tracking_eval/pred.csv"
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# Expresión regular para extraer el frame_id
frame_pattern = re.compile(r"video\d+_(\d+)_")

with open(INPUT_FILE, "r") as infile, open(OUTPUT_CSV, "w", newline="") as outfile:
    writer = csv.writer(outfile)
    for line in infile:
        parts = line.strip().split()
        if len(parts) < 2 or parts[1] == "NA":
            continue  # Skip invalid or missing predictions

        filename = parts[0]
        x, y, w, h = map(float, parts[1:])

        match = frame_pattern.search(filename)
        if match:
            frame_id = int(match.group(1))  # Extract frame number
            writer.writerow([frame_id, 0, x, y, w, h])
