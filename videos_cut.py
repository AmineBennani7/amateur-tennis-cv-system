import os
from moviepy.editor import VideoFileClip

input_path = "data/videos/video3.mp4"
output_path = "data/cropped_videos/video3_cropped.mp4"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

start_time = 50  # start second
end_time = 59    # end second

clip = VideoFileClip(input_path).subclip(start_time, end_time)

clip.write_videofile(output_path, codec="libx264", audio=True)

print(f"✅ Video recortado por tiempo guardado en {output_path}")
