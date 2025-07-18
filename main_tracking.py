import os
import cv2
import numpy as np
from ultralytics import YOLO
from trackers.predict_utils import getPredictions  # Returns list of (x,y)

import pandas as pd
from scipy.signal import find_peaks

from mini_court.mini_court import draw_minicourt  

VIDEO_PATH = "data/cropped_videos/video2_cropped.mp4"
OUTPUT_PATH = "outputs/video2_cropped_with_features_hits.mp4"  # Final single video with all features
IMGS_PER_BATCH = 5

os.makedirs("outputs", exist_ok=True)

model = YOLO("model/player_detector.pt")
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

# To store ALL the Y positions of the ball throughout the video
ball_positions_all = []

def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

print("Processing video and detecting players and ball...")

# Prepare mini court image once
mini_map_w, mini_map_h = 150, 200
mini_map_x, mini_map_y = frame_width - mini_map_w - 20, 20
mini_court_img = draw_minicourt(mini_map_w, mini_map_h)

# --- FIRST STEP: Collect all ball positions to detect hits later
while True:
    ret, frame = cap.read()
    if not ret:
        break

    original_frames.append(frame.copy())
    frames.append(frame)

    if len(frames) == IMGS_PER_BATCH:
        # Get ball positions for these frames
        ball_positions = getPredictions(original_frames, isBGRFormat=True)  # List of (x,y)
        ball_positions_all.extend(ball_positions)

        # Detect players and save bounding boxes
        for i in range(IMGS_PER_BATCH):
            result = model.track(
                source=frames[i],
                persist=True,
                tracker="bytetrack.yaml",
                stream=True,
                verbose=False
            )
            r = next(result)

            player_boxes = []
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    player_boxes.append((x1, y1, x2, y2))

            global_frame_num = batch_index * IMGS_PER_BATCH + i
            player_positions_by_frame[global_frame_num] = player_boxes

        frames = []
        original_frames = []
        batch_index += 1
        print(f"Processing batch {batch_index}/{total_frames//IMGS_PER_BATCH}", end="\r")

cap.release()

# --- SECOND STEP: Detect hits based on all ball Y positions

print("\nDetecting hits in ball Y position signal...")

y_positions = [pos[1] if pos[1] > 0 else -1 for pos in ball_positions_all]
df_ball_positions = pd.DataFrame({'y1': y_positions})
df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['y1'].rolling(window=5, min_periods=1).mean()
signal = df_ball_positions['mid_y_rolling_mean'].values

MIN_DISTANCE = 25
MIN_PROMINENCE = 130

peaks_max, _ = find_peaks(signal, distance=MIN_DISTANCE, prominence=MIN_PROMINENCE)
peaks_min, _ = find_peaks(-signal, distance=MIN_DISTANCE, prominence=MIN_PROMINENCE)
all_peaks = sorted(list(peaks_max) + list(peaks_min))

def group_close_peaks(peaks, signal, max_frame_gap=20):
    if not peaks:
        return []
    grouped = []
    current_group = [peaks[0]]
    for p in peaks[1:]:
        if p - current_group[-1] <= max_frame_gap:
            current_group.append(p)
        else:
            best_peak = max(current_group, key=lambda x: abs(signal[x]))
            grouped.append(best_peak)
            current_group = [p]
    best_peak = max(current_group, key=lambda x: abs(signal[x]))
    grouped.append(best_peak)
    return grouped

final_hits = group_close_peaks(all_peaks, signal, max_frame_gap=20)

def filter_small_vertical_jumps(peaks, signal, min_vertical_jump=100):
    if not peaks:
        return []
    filtered = [peaks[0]]
    for i in range(1, len(peaks)):
        prev_peak = filtered[-1]
        curr_peak = peaks[i]
        vertical_jump = abs(signal[curr_peak] - signal[prev_peak])
        if vertical_jump >= min_vertical_jump:
            filtered.append(curr_peak)
    return filtered

final_hits_filtered = filter_small_vertical_jumps(final_hits, signal, min_vertical_jump=120)

print(f"Hits detected at frames: {final_hits_filtered}")

# --- THIRD STEP: Reopen video and write video with ALL features and hit counter

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

    # Draw players on mini court
    if frame_number in player_positions_by_frame:
        for pid, bbox in enumerate(player_positions_by_frame[frame_number]):
            cx, cy = get_center(bbox)
            mini_x_pos = int(mini_map_x + (cx / frame_width) * mini_map_w)
            mini_y_pos = int(mini_map_y + (cy / frame_height) * mini_map_h)
            color = (255, 0, 0) if pid == 0 else (0, 0, 255)
            cv2.circle(frame, (mini_x_pos, mini_y_pos), 7, color, -1)
            cv2.putText(frame, f"P{pid}", (mini_x_pos - 10, mini_y_pos - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw ball on mini court
    if frame_number < len(ball_positions_all):
        x, y = ball_positions_all[frame_number]
        if x > 0 and y > 0:
            mini_x_pos = int(mini_map_x + (x / frame_width) * mini_map_w)
            mini_y_pos = int(mini_map_y + (y / frame_height) * mini_map_h)
            cv2.circle(frame, (mini_x_pos, mini_y_pos), 5, (0, 255, 255), -1)

    # Progress bar
    progress_width = frame_width - 40
    progress_height = 10
    progress_x = 20
    progress_y = frame_height - 30
    cv2.rectangle(frame, (progress_x, progress_y),
                  (progress_x + progress_width, progress_y + progress_height), (100, 100, 100), -1)
    progress_filled = int(progress_width * (frame_number / total_frames))
    cv2.rectangle(frame, (progress_x, progress_y),
                  (progress_x + progress_filled, progress_y + progress_height), (0, 255, 0), -1)

    # Frame number
    text = f"Frame: {frame_number}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = (frame_width - text_size[0]) // 2
    text_y = 30
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 255), thickness)

    # Hit counter
    while hit_index < len(final_hits_filtered) and frame_number >= final_hits_filtered[hit_index]:
        hit_count += 1
        hit_index += 1

    cv2.putText(frame, f"Hits detected: {hit_count}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    video_writer.write(frame)
    frame_number += 1

cap.release()
video_writer.release()
cv2.destroyAllWindows()

print(f"Final video saved at: {OUTPUT_PATH}")
