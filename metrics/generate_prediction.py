import os
import sys
import cv2
import numpy as np
from collections import defaultdict

# Add path to the 'predict_utils' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from trackers.predict_utils import getPredictions

# Configuration
image_folder = "metrics/tennis_ball_eval_dataset/images"
output_file = "metrics/predictions/pred_ball_coords.txt"
os.makedirs("metrics/predictions", exist_ok=True)

IMGS_PER_INSTANCE = 5  # Number of consecutive frames needed for prediction
FIXED_BBOX_SIZE = 0.03  # Normalized bounding box size for ball

# Group images by clip name prefix 
groups = defaultdict(list)
for fname in os.listdir(image_folder):
    if fname.endswith(".jpg"):
        prefix = fname.split("_")[0]
        groups[prefix].append(fname)

total_preds = 0

# Open the output file to store predictions
with open(output_file, "w") as f_out:
    for group, files in groups.items():
        # Sort frames numerically based on their frame number
        files_sorted = sorted(files, key=lambda x: int(x.split("_")[1]))
        frame_buffer = []

        print(f"Processing clip: {group} with {len(files_sorted)} images...")

        for fname in files_sorted:
            img_path = os.path.join(image_folder, fname)
            image = cv2.imread(img_path)

            if image is None:
                print(f"Could not load image: {img_path}")
                continue

            frame_buffer.append(image)

            # Once enough frames are gathered, make a prediction
            if len(frame_buffer) == IMGS_PER_INSTANCE:
                try:
                    preds = getPredictions(frame_buffer, isBGRFormat=True)
                    x, y = preds[-1]  # Use the prediction for the last frame in the buffer

                    h, w = image.shape[:2]
                    x_norm = x / w
                    y_norm = y / h

                    if x == 0 and y == 0:
                        print(f"No detection in {fname}")
                        # log a "no detection" event
                        f_out.write(f"{fname} NA NA NA NA\n")
                    else:
                        print(f"Detection in {fname}: ({x_norm:.3f}, {y_norm:.3f})")
                        f_out.write(f"{fname} {x_norm:.6f} {y_norm:.6f} {FIXED_BBOX_SIZE:.6f} {FIXED_BBOX_SIZE:.6f}\n")
                        total_preds += 1

                except Exception as e:
                    print(f"Error processing {fname}: {e}")
                    f_out.write(f"{fname} NA NA NA NA\n")

                # Slide the frame window by one
                frame_buffer.pop(0)

print("Prediction complete.")
print(f"Total valid detections with coordinates: {total_preds}")
