import os
import cv2
import numpy as np
from ultralytics import YOLO
from trackers.predict_utils import getPredictions  # Assumes this function returns ball (x, y) positions

VIDEO_PATH = "data/cropped_videos/video2_cropped.mp4"
OUTPUT_PATH = "outputs/video2_cropped_with_features.mp4"
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

def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    original_frames.append(frame.copy())
    frames.append(frame)

    if len(frames) == IMGS_PER_BATCH:
        base_frame_index = batch_index * IMGS_PER_BATCH

        # Get ball positions for these frames
        ball_positions = getPredictions(original_frames, isBGRFormat=True)  # Returns list of (x, y) tuples

        # Detect players and store bounding boxes per frame
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

            global_frame_num = base_frame_index + i
            player_positions_by_frame[global_frame_num] = player_boxes

        # Draw detections and overlays for each frame
        for i in range(IMGS_PER_BATCH):
            frame_tracked = frames[i].copy()
            global_frame_num = base_frame_index + i

            # Draw player bounding boxes
            if global_frame_num in player_positions_by_frame:
                for bbox in player_positions_by_frame[global_frame_num]:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame_tracked, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Draw ball on frame
            x, y = ball_positions[i]
            if x > 0 and y > 0:
                cv2.circle(frame_tracked, (x, y), 6, (0, 0, 255), 3)  # Red circle for ball

            # Mini map parameters and background
            mini_map_w, mini_map_h = 150, 200
            mini_map_x, mini_map_y = frame_width - mini_map_w - 20, 20
            cv2.rectangle(frame_tracked, (mini_map_x, mini_map_y),
                          (mini_map_x + mini_map_w, mini_map_y + mini_map_h), (50, 50, 50), -1)

            # Draw player centers on mini map
            if global_frame_num in player_positions_by_frame:
                for pid, bbox in enumerate(player_positions_by_frame[global_frame_num]):
                    cx, cy = get_center(bbox)
                    mini_x = int(mini_map_x + (cx / frame_width) * mini_map_w)
                    mini_y = int(mini_map_y + (cy / frame_height) * mini_map_h)
                    color = (255, 0, 0) if pid == 0 else (0, 0, 255)
                    cv2.circle(frame_tracked, (mini_x, mini_y), 7, color, -1)
                    cv2.putText(frame_tracked, f"P{pid}", (mini_x - 10, mini_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw ball on mini map
            if x > 0 and y > 0:
                mini_x = int(mini_map_x + (x / frame_width) * mini_map_w)
                mini_y = int(mini_map_y + (y / frame_height) * mini_map_h)
                cv2.circle(frame_tracked, (mini_x, mini_y), 5, (0, 255, 255), -1)  # Cyan circle for ball on mini map

            # Progress bar
            progress_width = frame_width - 40
            progress_height = 10
            progress_x = 20
            progress_y = frame_height - 30
            cv2.rectangle(frame_tracked, (progress_x, progress_y),
                          (progress_x + progress_width, progress_y + progress_height), (100, 100, 100), -1)
            progress_filled = int(progress_width * (global_frame_num / total_frames))
            cv2.rectangle(frame_tracked, (progress_x, progress_y),
                          (progress_x + progress_filled, progress_y + progress_height), (0, 255, 0), -1)

            video_writer.write(frame_tracked)

        frames = []
        original_frames = []

    batch_index += 1
    print(f"Processing frame batch {batch_index}/{total_frames//IMGS_PER_BATCH}", end="\r")

cap.release()
video_writer.release()
cv2.destroyAllWindows()
print(f"\nVideo saved to: {OUTPUT_PATH}")
