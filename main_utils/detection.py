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
    global last_valid_players, missing_counter

    # Get ball positions using a separate predictor
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

        bboxes = []
        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bboxes.append((x1, y1, x2, y2))

        # Remove duplicates due to very close bounding boxes (false positives or overlapping)
        filtered_bboxes = []
        for box in bboxes:
            c = get_center(box)
            too_close = False
            for fb in filtered_bboxes:
                if np.linalg.norm(c - get_center(fb)) < 30:  # pixels
                    too_close = True
                    break
            if not too_close:
                filtered_bboxes.append(box)

        # Sort boxes by Y coordinate: lower on screen (closer to camera) is player 0
        filtered_bboxes = sorted(filtered_bboxes, key=lambda b: get_center(b)[1])

        players_this_frame = [None, None]

        # Assign boxes if available
        if len(filtered_bboxes) >= 1:
            players_this_frame[0] = filtered_bboxes[-1]  # bottom player
            last_valid_players[0] = players_this_frame[0]
            missing_counter[0] = 0
        else:
            if missing_counter[0] < max_persistence and last_valid_players[0] is not None:
                players_this_frame[0] = last_valid_players[0]
                missing_counter[0] += 1

        if len(filtered_bboxes) >= 2:
            players_this_frame[1] = filtered_bboxes[0]  # top player
            last_valid_players[1] = players_this_frame[1]
            missing_counter[1] = 0
        else:
            if missing_counter[1] < max_persistence and last_valid_players[1] is not None:
                players_this_frame[1] = last_valid_players[1]
                missing_counter[1] += 1

        # Save valid results for this frame
        global_frame_num = batch_index * imgs_per_batch + i
        valid_boxes = [p for p in players_this_frame if p is not None]
        player_positions_by_frame[global_frame_num] = valid_boxes
