import cv2
import os

# Folder containing your videos
video_folder = 'data/videos'
# Folder where extracted frames will be saved
output_folder = 'data/frames'
# Frame extraction interval (in seconds)
frame_interval = 4  # extract one frame every 4 seconds

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate over all .webm files in the video folder
for video_file in os.listdir(video_folder):
    if video_file.endswith('.mp4'):
        video_path = os.path.join(video_folder, video_file)
        cap = cv2.VideoCapture(video_path)

        # Create a subfolder for the frames of this video
        video_name = os.path.splitext(video_file)[0]
        video_output_dir = os.path.join(output_folder, video_name)
        os.makedirs(video_output_dir, exist_ok=True)

        fps = cap.get(cv2.CAP_PROP_FPS)
        interval_frames = int(fps * frame_interval)
        frame_count = 0
        saved_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % interval_frames == 0:
                frame_filename = os.path.join(video_output_dir, f"frame_{saved_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                saved_count += 1
            frame_count += 1

        cap.release()
        print(f"✔️ {saved_count} frames saved from {video_file} in {video_output_dir}")

print(" Frame extraction completed.")
