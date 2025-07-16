import os
import csv
import re
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

LABELS_DIR = "metrics/tennis_players_eval_dataset/labels"
OUTPUT_CSV = "metrics/mot_metrics/player_metrics/tracking_eval/gt.csv"
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# Regular expression to extract frame ID from filename
frame_pattern = re.compile(r"^(video\d+_\d+_jpg\.rf\.[a-zA-Z0-9]+)\.txt$")

with open(OUTPUT_CSV, "w", newline="") as outfile:
    writer = csv.writer(outfile)

    for file in os.listdir(LABELS_DIR):
        if not file.endswith(".txt"):
            continue

        match = frame_pattern.match(file)
        if not match:
            print(f"No match: {file}")
            continue

        frame_id = match.group(1)

        with open(os.path.join(LABELS_DIR, file), "r") as f:
            boxes = []
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x, y, w, h = map(float, parts)
                if int(cls) != 0:
                    continue  # skip non-player classes
                boxes.append((x, y, w, h))

            # Sort by bounding box height (descending) to prioritize closer players
            boxes_sorted = sorted(boxes, key=lambda b: b[3], reverse=True)

            for i, (x, y, w, h) in enumerate(boxes_sorted[:2]):
                writer.writerow([frame_id, i, x, y, w, h])
                print(f"GT: Frame {frame_id}, ID {i}")
