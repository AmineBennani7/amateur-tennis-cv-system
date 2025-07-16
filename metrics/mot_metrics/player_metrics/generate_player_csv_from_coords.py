import os
import csv
import re

# File paths
INPUT_FILE = "metrics/predictions/pred_player_coords.txt"
OUTPUT_CSV = "metrics/mot_metrics/player_metrics/tracking_eval/pred.csv"
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# Pattern to extract the unique frame identifier from the image filename

frame_pattern = re.compile(r"^(video\d+_\d+_jpg\.rf\.[a-zA-Z0-9]+)\.jpg$")

with open(INPUT_FILE, "r") as infile, open(OUTPUT_CSV, "w", newline="") as outfile:
    writer = csv.writer(outfile)

    for line in infile:
        parts = re.split(r'\s+', line.strip())
        if len(parts) != 6:
            print(f"❌ Skipping invalid line: {line.strip()}")
            continue

        filename = parts[0]
        match = frame_pattern.match(filename)
        if not match:
            print(f"Filename did not match pattern: {filename}")
            continue

        frame_id = match.group(1)  # Use same ID format as in gt.csv
        player_id = int(parts[1])
        x, y, w, h = map(float, parts[2:])

        writer.writerow([frame_id, player_id, x, y, w, h])
        print(f"Wrote: {frame_id}, {player_id}, {x}, {y}, {w}, {h}")
