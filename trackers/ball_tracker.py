from keras.models import load_model
from gridtrack_model import GridTrackNet  
from predict_utils import getPredictions    
import cv2
import os

# Path to the input video
VIDEO_PATH = "data/cropped_videos/video1_cropped.mp4"

# Create output folder if it doesn't exist
output_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
os.makedirs(output_folder, exist_ok=True)

# Output file name and path
output_filename = "ball_tracking_output.mp4"
output_path = os.path.join(output_folder, output_filename)

# Open video capture
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error opening video file: {VIDEO_PATH}")
    exit()

# Video parameters
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# Initialize video writer
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

frames = []
ballCoordinatesHistory = []
index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frames.append(frame)
    if len(frames) == 5:
        predictions = getPredictions(frames, isBGRFormat=True)
        for i, frame in enumerate(frames):
            x, y = predictions[i]
            if x > 0 and y > 0:
                cv2.circle(frame, (x, y), 6, (0, 0, 255), 3)
                ballCoordinatesHistory.append((x, y))
            video_writer.write(frame)
        frames = []

    index += 1
    print(f"Processing frame {index}", end="\r")

# Finalize
cap.release()
video_writer.release()
cv2.destroyAllWindows()
print("\nTracking completed. Video saved as:", output_path)
