from ultralytics import YOLO
from trackers.predict_utils import getPredictions
import numpy as np

def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

# Store last known valid positions for each player (player 0 and 1)
last_valid_players = {0: None, 1: None}
# Count how many consecutive frames a player has been missing
missing_counter = {0: 0, 1: 0}
# Maximum number of frames a player can be missing while maintaining their last known position
max_persistence = 20

def detect_players_and_ball(model, frames, original_frames, batch_index, player_positions_by_frame, ball_positions_all, imgs_per_batch):
    # Get ball positions
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

        tracked_players = {}

        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id != 0:  # class 0 is the player from our robotflow dataset
                    continue
                track_id = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                tracked_players[track_id] = (x1, y1, x2, y2)

        # Save all tracked player boxes (with their real IDs)
        global_frame_num = batch_index * imgs_per_batch + i
        player_positions_by_frame[global_frame_num] = tracked_players
