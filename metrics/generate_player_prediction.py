import os
import cv2
import numpy as np
from ultralytics import YOLO
import re

# SETTINGS
IMAGES_DIR = "metrics/tennis_players_eval_dataset/images"
OUTPUT_PATH = "metrics/predictions/pred_player_coords.txt"
MODEL_FILE = "model/player_detector.pt"
MIN_WIDTH = 0.01
MIN_HEIGHT = 0.03

# Regex to extract frame number from filename
frame_pattern = re.compile(r"match\d+_(\d+)_png\.rf")

# Make sure the output directory exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Load the YOLO model with ByteTrack enabled
model = YOLO(MODEL_FILE)

# Group images by clip prefix (e.g., match1_000X)
groups = {}
for filename in os.listdir(IMAGES_DIR):
    if filename.endswith(".jpg"):
        prefix = "_".join(filename.split("_")[:2])
        groups.setdefault(prefix, []).append(filename)

total_predictions = 0

with open(OUTPUT_PATH, "w") as out_file:
    for group, files in groups.items():
        print(f"Processing clip: {group} ({len(files)} frames)")

        # Sort frames numerically by frame number
        sorted_files = sorted(
            files,
            key=lambda name: int(frame_pattern.search(name).group(1)) if frame_pattern.search(name) else -1
        )

        for filename in sorted_files:
            image_path = os.path.join(IMAGES_DIR, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue

            # Run ByteTrack tracking
            results = model.track(
                source=image,
                persist=True,
                tracker="bytetrack.yaml",
                stream=True,
                verbose=False
            )

            result = next(results)
            h, w = image.shape[:2]

            if result.boxes is not None:
                for box in result.boxes:
                    if not hasattr(box, 'id') or box.id is None:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    track_id = int(box.id.item())

                    # Normalize box dimensions and skip if too small
                    box_width = (x2 - x1) / w
                    box_height = (y2 - y1) / h
                    if box_width < MIN_WIDTH or box_height < MIN_HEIGHT:
                        continue

                    # Calculate normalized center coordinates
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h

                    # Write prediction to output
                    out_file.write(f"{filename} {track_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
                    total_predictions += 1

print("Finished generating predictions.")
print(f"Total predictions saved: {total_predictions}")
