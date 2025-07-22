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

# --- Configuration ---
VIDEO_PATH = "data/cropped_videos/video3_cropped.mp4"
OUTPUT_PATH = "outputs/video3_cropped_with_features_hits.mp4"
IMGS_PER_BATCH = 5

os.makedirs("outputs", exist_ok=True)
os.makedirs("calibration", exist_ok=True)

model = YOLO("model/player_detector.pt")

# --- Role assignment tracking ---
assigned_roles = {}  # track_id → "A" or "B"
role_to_track = {}   # "A"/"B" → track_id

def assign_player_roles(current_boxes, H):
    """
    Assign consistent logical roles A/B to players across frames.

    Args:
        current_boxes: dict {track_id: bbox}
        H: homography matrix

    Returns:
        dict {logical_id: (track_id, bbox)}
    """
    global assigned_roles, role_to_track
    roles_output = {}

    # Step 1: Keep existing assignments if track_id is still visible
    for track_id, bbox in current_boxes.items():
        if track_id in assigned_roles:
            role = assigned_roles[track_id]
            roles_output[role] = (track_id, bbox)

    # Step 2: Assign remaining players by X position (left = A)
    remaining = {tid: bbox for tid, bbox in current_boxes.items() if tid not in assigned_roles}
    if len(remaining) > 0:
        projected = []
        for tid, box in remaining.items():
            cx, cy = get_center(box)
            try:
                proj_x, proj_y = apply_homography((cx, cy), H)
                projected.append((proj_x, tid, box))
            except:
                continue

        projected.sort(key=lambda x: x[0])  # left to right

        for proj_x, tid, box in projected:
            if "A" not in roles_output and "A" not in role_to_track:
                roles_output["A"] = (tid, box)
                assigned_roles[tid] = "A"
                role_to_track["A"] = tid
            elif "B" not in roles_output and "B" not in role_to_track:
                roles_output["B"] = (tid, box)
                assigned_roles[tid] = "B"
                role_to_track["B"] = tid

    # Step 3: Cleanup roles if a track disappeared
    current_ids = set(current_boxes.keys())
    lost_ids = [tid for tid in assigned_roles if tid not in current_ids]
    for tid in lost_ids:
        role = assigned_roles[tid]
        del assigned_roles[tid]
        if role_to_track.get(role) == tid:
            del role_to_track[role]

    return roles_output


# --- Calibrate homography ---
print("Calibrating homography... Select the 4 court corners in the window.")
H = calibrate_homography(VIDEO_PATH)

# --- Load video metadata ---
cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

# --- Minimap configuration ---
mini_map_w, mini_map_h = 150, 200
mini_map_x, mini_map_y = frame_width - mini_map_w - 20, 20
mini_court_img = draw_minicourt(mini_map_w, mini_map_h)

frames = []
original_frames = []
batch_index = 0
player_positions_by_frame = {}
ball_positions_all = []
last_proj_ball_pos = None
alpha = 0.3

# --- Step 1: Detect players and ball ---
print("Processing video and detecting players and ball...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    original_frames.append(frame.copy())
    frames.append(frame)

    if len(frames) == IMGS_PER_BATCH:
        detect_players_and_ball(
            model, frames, original_frames, batch_index,
            player_positions_by_frame, ball_positions_all, IMGS_PER_BATCH
        )
        frames, original_frames = [], []
        batch_index += 1
        print(f"Processing batch {batch_index}/{total_frames // IMGS_PER_BATCH}", end="\r")

cap.release()

# --- Step 2: Detect hits based on ball Y positions ---
print("\nDetecting hits in ball Y position signal...")
final_hits_filtered = detect_hits(ball_positions_all)
print(f"Hits detected at frames: {final_hits_filtered}")

# --- Step 3: Draw final video with overlay ---
cap = cv2.VideoCapture(VIDEO_PATH)
frame_number = 0
hit_index = 0
hit_count = 0

print("Generating final video with detections and hit count...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Assign roles for current frame
    roles = assign_player_roles(player_positions_by_frame.get(frame_number, {}), H)
    trackid_to_role = {tid: role for role, (tid, _) in roles.items()}

    # Draw players with consistent logical roles
    if frame_number in player_positions_by_frame:
        for track_id, bbox in player_positions_by_frame[frame_number].items():
            x1, y1, x2, y2 = bbox
            role = trackid_to_role.get(track_id, "")
            if role == "":
                continue

            color = (255, 0, 0) if role == "A" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, role, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Draw ball
    if frame_number < len(ball_positions_all):
        x, y = ball_positions_all[frame_number]
        if x > 0 and y > 0:
            cv2.circle(frame, (x, y), 6, (0, 0, 255), 3)

    # Draw minimap background
    frame[mini_map_y:mini_map_y + mini_map_h, mini_map_x:mini_map_x + mini_map_w] = mini_court_img.copy()

    # Draw players on minimap
    for role, (track_id, bbox) in roles.items():
        cx, cy = get_center(bbox)
        try:
            proj_x, proj_y = apply_homography((cx, cy), H)
            proj_x = np.clip(proj_x, 0, 10.97)
            proj_y = np.clip(proj_y, 0, 11.89)
            mini_x = int(mini_map_x + (proj_x / 10.97) * mini_map_w)
            mini_y = int(mini_map_y + (proj_y / 11.89) * mini_map_h)
            color = (255, 0, 0) if role == "A" else (0, 0, 255)
            cv2.circle(frame, (mini_x, mini_y), 6, color, -1)
            cv2.putText(frame, role, (mini_x - 10, mini_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        except:
            pass

    # Draw ball on minimap
    if frame_number < len(ball_positions_all):
        x, y = ball_positions_all[frame_number]
        if x > 0 and y > 0:
            try:
                proj_x, proj_y = apply_homography((x, y), H)
                proj_x = np.clip(proj_x, 0, 10.97)
                proj_y = np.clip(proj_y, 0, 11.89)
                if last_proj_ball_pos is None:
                    last_proj_ball_pos = (proj_x, proj_y)
                else:
                    last_proj_ball_pos = (
                        alpha * proj_x + (1 - alpha) * last_proj_ball_pos[0],
                        alpha * proj_y + (1 - alpha) * last_proj_ball_pos[1]
                    )
                mini_x = int(mini_map_x + (last_proj_ball_pos[0] / 10.97) * mini_map_w)
                mini_y = int(mini_map_y + (last_proj_ball_pos[1] / 11.89) * mini_map_h)
                cv2.circle(frame, (mini_x, mini_y), 5, (0, 255, 255), -1)
            except:
                pass

    # Progress bar
    bar_w = frame_width - 40
    filled = int(bar_w * (frame_number / total_frames))
    cv2.rectangle(frame, (20, frame_height - 30), (20 + bar_w, frame_height - 20), (100, 100, 100), -1)
    cv2.rectangle(frame, (20, frame_height - 30), (20 + filled, frame_height - 20), (0, 255, 0), -1)

    # Frame number
    cv2.putText(frame, f"Frame: {frame_number}", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

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
os.remove("calibration/homography_matrix.npy")

print(f"✅ Final video saved at: {OUTPUT_PATH}")
