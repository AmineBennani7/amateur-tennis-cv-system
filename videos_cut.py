import os
from moviepy.editor import VideoFileClip

#  Define the path to the original video 
input_path = "data/videos/video1.webm"

# Define the output folder and create it if it doesn't exist =
output_folder = "data/cropped_videos"
os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist
# Load the original video 
clip = VideoFileClip(input_path)

# =Define the segment to extract: from 30s to 45s 
subclip = clip.subclip(30, 45)  # You can adjust these values

# Define the output file path 
output_path = os.path.join(output_folder, "video1_cropped.mp4")

# Write the cropped video to file
subclip.write_videofile(output_path, codec="libx264")  # Save the new clip
