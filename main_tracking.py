import os
import cv2
import numpy as np
from ultralytics import YOLO
from trackers.predict_utils import getPredictions
import pandas as pd

from main_utils.calibration import calibrate_homography, SCALE, REFERENCE_POINTS
from main_utils.detection import detect_players_and_ball, get_center
from main_utils.hit_detection import detect_hits
from main_utils.visualisation import draw_minicourt, apply_homography

VIDEO_PATH = "data/cropped_videos/video3_cropped.mp4"
OUTPUT_PATH = "outputs/video3_cropped_with_features_hits.mp4"
IMGS_PER_BATCH = 5

os.makedirs("outputs", exist_ok=True)
os.makedirs("calibration", exist_ok=True)

model = YOLO("model/player_detector.pt")

print("Calibrating homography... Select the 4 court corners in the window.")
H = calibrate_homography(VIDEO_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

frames = []
original_frames = []
batch_index = 0
player_positions_by_frame = {}
ball_positions_all = []

# Prepare mini court image once
mini_map_w, mini_map_h = 150, 200
mini_map_x, mini_map_y = frame_width - mini_map_w - 20, 20
mini_court_img = draw_minicourt(mini_map_w, mini_map_h)

last_proj_ball_pos = None
alpha = 0.3  # smoothing coefficient

print("Processing video and detecting players and ball...")

# --- STEP 1: Collect ball positions and detect players ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    original_frames.append(frame.copy())
    frames.append(frame)

    if len(frames) == IMGS_PER_BATCH:
        detect_players_and_ball(model, frames, original_frames, batch_index, player_positions_by_frame, ball_positions_all, IMGS_PER_BATCH)

        frames = []
        original_frames = []
        batch_index += 1
        print(f"Processing batch {batch_index}/{total_frames//IMGS_PER_BATCH}", end="\r")

cap.release()

# --- STEP 2: Detect hits based on ball Y position signal ---
print("\nDetecting hits in ball Y position signal...")
final_hits_filtered = detect_hits(ball_positions_all)
print(f"Hits detected at frames: {final_hits_filtered}")

# --- STEP 3: Reopen video and write output with all features ---
cap = cv2.VideoCapture(VIDEO_PATH)
frame_number = 0
hit_index = 0
hit_count = 0

print("Generating final video with detection and hit count...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw player bounding boxes
    if frame_number in player_positions_by_frame:
        for bbox in player_positions_by_frame[frame_number]:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Draw ball
    if frame_number < len(ball_positions_all):
        x, y = ball_positions_all[frame_number]
        if x > 0 and y > 0:
            cv2.circle(frame, (x, y), 6, (0, 0, 255), 3)

    # Draw mini court background image
    frame[mini_map_y:mini_map_y + mini_map_h, mini_map_x:mini_map_x + mini_map_w] = mini_court_img

    # Draw players on mini court with homography projection
    if frame_number in player_positions_by_frame:
        for pid, bbox in enumerate(player_positions_by_frame[frame_number]):
            cx, cy = get_center(bbox)
            proj_x, proj_y = apply_homography((cx, cy), H)
            mini_x_pos = int(mini_map_x + (proj_x / 10.97) * mini_map_w)
            mini_y_pos = int(mini_map_y + (proj_y / 11.89) * mini_map_h)
            color = (255, 0, 0) if pid == 0 else (0, 0, 255)
            cv2.circle(frame, (mini_x_pos, mini_y_pos), 7, color, -1)
            cv2.putText(frame, f"P{pid}", (mini_x_pos - 10, mini_y_pos - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw ball on mini court with homography projection and smoothing + clipping
    if frame_number < len(ball_positions_all):
        x, y = ball_positions_all[frame_number]
        if x > 0 and y > 0:
            proj_x, proj_y = apply_homography((x, y), H)

            # Clip projected position inside court bounds
            proj_x = np.clip(proj_x, 0, 10.97)
            proj_y = np.clip(proj_y, 0, 11.89)

            if last_proj_ball_pos is None:
                last_proj_ball_pos = (proj_x, proj_y)
            else:
                last_proj_ball_pos = (
                    alpha * proj_x + (1 - alpha) * last_proj_ball_pos[0],
                    alpha * proj_y + (1 - alpha) * last_proj_ball_pos[1]
                )

            clipped_x = np.clip(last_proj_ball_pos[0], 0, 10.97)
            clipped_y = np.clip(last_proj_ball_pos[1], 0, 11.89)

            mini_x_pos = int(mini_map_x + (clipped_x / 10.97) * mini_map_w)
            mini_y_pos = int(mini_map_y + (clipped_y / 11.89) * mini_map_h)
            cv2.circle(frame, (mini_x_pos, mini_y_pos), 5, (0, 255, 255), -1)

    # Progress bar
    progress_width = frame_width - 40
    progress_height = 10
    progress_x = 20
    progress_y = frame_height - 30
    cv2.rectangle(frame, (progress_x, progress_y), (progress_x + progress_width, progress_y + progress_height), (100, 100, 100), -1)
    progress_filled = int(progress_width * (frame_number / total_frames))
    cv2.rectangle(frame, (progress_x, progress_y), (progress_x + progress_filled, progress_y + progress_height), (0, 255, 0), -1)

    # Frame number display
    text = f"Frame: {frame_number}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = (frame_width - text_size[0]) // 2
    text_y = 30
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 255), thickness)

    # Hit counter update and display
    while hit_index < len(final_hits_filtered) and frame_number >= final_hits_filtered[hit_index]:
        hit_count += 1
        hit_index += 1
    cv2.putText(frame, f"Hits detected: {hit_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    video_writer.write(frame)
    frame_number += 1

cap.release()
video_writer.release()
cv2.destroyAllWindows()

# Delete calibration file after processing
os.remove("calibration/homography_matrix.npy")
print("Calibration file deleted after finishing video.")

print(f"Final video saved at: {OUTPUT_PATH}")
