# trackers/player_tracker.py

import cv2
from ultralytics import YOLO

class PlayerTracker:
    def __init__(self, model_path):
        """
        Initialize the tracker by loading the YOLO model.
        :param model_path: path to a YOLOv8 model (e.g., yolov8n.pt or a custom-trained model)
        """
        self.model = YOLO(model_path)
        self.class_id = 0  # Class ID "player" (depends on your training)

    def detect_frame(self, frame):
        """
        Detects players (class_id = 1) in a given frame using YOLO.
        :param frame: a single BGR image (frame from a video)
        :return: a dictionary with {track_id: bounding_box}, where bounding_box = (x1, y1, x2, y2)
        """
        results = self.model.predict(frame, verbose=False)[0]
        detections = {}

        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls[0])
            if cls_id != self.class_id:
                continue  # Skip other classes

            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            detections[i] = (x1, y1, x2, y2)  # Assign box to a track ID (placeholder: i)

        return detections

    def draw_bboxes(self, frames, player_detections):
        """
        Draws bounding boxes and track IDs on each frame.
        :param frames: list of frames (images)
        :param player_detections: list of dictionaries with detections per frame
        :return: list of frames with visual annotations (bounding boxes + IDs)
        """
        output = []

        for frame, player_dict in zip(frames, player_detections):
            if not player_dict:
                output.append(frame)
                continue  # Skip if no detections

            for track_id, (x1, y1, x2, y2) in player_dict.items():
                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Add track ID as text
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            output.append(frame)

        return output
