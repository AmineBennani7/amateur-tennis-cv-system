
import os
import cv2
import numpy as np
from ultralytics import YOLO
from trackers.predict_utils import getPredictions

# Paths
VIDEO_PATH = "data/cropped_videos/video1_cropped.mp4"
OUTPUT_PATH = "outputs/video1_cropped.mp4"
IMGS_PER_INSTANCE = 5

os.makedirs("outputs", exist_ok=True)

model = YOLO("model/player_detector.pt")
cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

frames = []
original_frames = []
index = 0
player_refs = {}

def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

def match_players_to_ids(current_boxes, player_refs):
    current_centers = [get_center(b) for b in current_boxes]
    matched_ids = [-1] * len(current_boxes)

    if len(current_boxes) == 2:
        # Siempre que haya exactamente 2 jugadores, reevaluamos
        sorted_by_height = sorted(
            enumerate(current_boxes), 
            key=lambda x: (x[1][3] - x[1][1]), 
            reverse=True
        )
        idx0 = sorted_by_height[0][0]  # más cercano → ID 0
        idx1 = sorted_by_height[1][0]  # más lejano → ID 1
        matched_ids[idx0] = 0
        matched_ids[idx1] = 1
        player_refs[0] = current_boxes[idx0]
        player_refs[1] = current_boxes[idx1]
    else:
        # Si no hay exactamente 2 jugadores, usamos matching por distancia
        ref_centers = {pid: get_center(bbox) for pid, bbox in player_refs.items()}
        for i, center in enumerate(current_centers):
            distances = {pid: np.linalg.norm(center - ref_centers[pid]) for pid in ref_centers}
            if distances:
                matched_pid = min(distances, key=distances.get)
                matched_ids[i] = matched_pid
                player_refs[matched_pid] = current_boxes[i]

    return matched_ids


while True:
    ret, frame = cap.read()
    if not ret:
        break

    original_frames.append(frame.copy())
    frames.append(frame)

    if len(frames) == IMGS_PER_INSTANCE:
        ball_positions = getPredictions(original_frames, isBGRFormat=True)

        for i in range(IMGS_PER_INSTANCE):
            result = model.track(
                source=frames[i],
                persist=True,
                tracker="bytetrack.yaml",
                stream=True,
                verbose=False
            )
            r = next(result)
            frame_tracked = frames[i].copy()
            player_boxes = []

            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    player_boxes.append((x1, y1, x2, y2))

                matched_ids = match_players_to_ids(player_boxes, player_refs)

                for (x1, y1, x2, y2), pid in zip(player_boxes, matched_ids):
                    if pid != -1:
                        cv2.rectangle(frame_tracked, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame_tracked, f"ID:{pid}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            x, y = ball_positions[i]
            if x > 0 and y > 0:
                cv2.circle(frame_tracked, (x, y), 6, (0, 0, 255), 3)

            video_writer.write(frame_tracked)

        frames = []
        original_frames = []

    index += 1
    print(f"Processing frame {index}", end="\r")

cap.release()
video_writer.release()
cv2.destroyAllWindows()
print(f"Final video saved at: {OUTPUT_PATH}")
