from ultralytics import YOLO
from trackers.predict_utils import getPredictions  # Returns list of (x,y)
import numpy as np

def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

def detect_players_and_ball(model, frames, original_frames, batch_index, player_positions_by_frame, ball_positions_all, imgs_per_batch):
    ball_positions = getPredictions(original_frames, isBGRFormat=True)
    ball_positions_all.extend(ball_positions)

    for i in range(imgs_per_batch):
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

        global_frame_num = batch_index * imgs_per_batch + i
        player_positions_by_frame[global_frame_num] = player_boxes
