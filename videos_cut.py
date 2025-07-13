import os
import cv2
from moviepy.editor import VideoFileClip

input_path = "data/videos/video3.mp4"
output_base_folder = "metrics/video3"
os.makedirs(output_base_folder, exist_ok=True)

# Dividir desde el segundo 60 hasta el final (300 s) en 3 clips
segments = [
    (60, 140),
    (140, 220),
    (220, 300)
]

save_every_n = 1
MAX_FRAMES = 64

for idx, (start, end) in enumerate(segments):
    clip = VideoFileClip(input_path).subclip(start, end)
    temp_video_path = f"temp_segment_{idx}.mp4"
    clip.write_videofile(temp_video_path, codec="libx264", audio=False, verbose=False, logger=None)

    cap = cv2.VideoCapture(temp_video_path)
    clip_folder = os.path.join(output_base_folder, f"Clip{idx}")
    os.makedirs(clip_folder, exist_ok=True)

    frame_index = 0
    saved_count = 0
    while cap.isOpened() and saved_count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % save_every_n == 0:
            frame_filename = os.path.join(clip_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
        frame_index += 1

    cap.release()
    os.remove(temp_video_path)
    print(f"✅ Clip{idx}: {saved_count} frames guardados en '{clip_folder}'")

print("📦 3 clips extraídos desde el minuto 1:00 hasta el final de video3, con máximo 64 frames cada uno.")
