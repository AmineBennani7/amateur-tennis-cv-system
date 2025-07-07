import cv2
import os
from trackers.player_tracker import PlayerTracker
from trackers.predict_utils import getPredictions

# Parameters
VIDEO_PATH = "data/videos/Faris Boubekri Vs Manuel Porras.f135.mp4"
OUTPUT_PATH = "outputs/final_output_2.mp4"
IMGS_PER_INSTANCE = 5

# Load models
player_tracker = PlayerTracker("model/player_detector.pt")

# Video capture
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error opening video: {VIDEO_PATH}")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

os.makedirs("outputs", exist_ok=True)
video_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

frames = []
original_frames = []
index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save original frame before any drawing
    original_frames.append(frame.copy())
    frames.append(frame)

    if len(frames) == IMGS_PER_INSTANCE:
        # Predict ball positions on original (unmodified) frames
        ball_positions = getPredictions(original_frames, isBGRFormat=True)

        for i in range(IMGS_PER_INSTANCE):
            # Draw player detections on each frame
            detections = player_tracker.detect_frame(frames[i])
            if detections:
                frames[i] = player_tracker.draw_bboxes([frames[i]], [detections])[0]

            # Draw ball prediction
            x, y = ball_positions[i]
            if x > 0 and y > 0:
                cv2.circle(frames[i], (x, y), 6, (0, 0, 255), 3)

            video_writer.write(frames[i])

        frames = []
        original_frames = []

    index += 1
    print(f"Processing frame {index}", end="\r")

cap.release()
video_writer.release()
cv2.destroyAllWindows()
print(f"\n✅ Final combined tracking saved to: {OUTPUT_PATH}")
