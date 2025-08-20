import os
from moviepy.editor import VideoFileClip

# Input and output directories
input_dir = "data/videos"
output_dir = "data/cropped_videos"
os.makedirs(output_dir, exist_ok=True)

# Dictionary with cut times for each video (in seconds)
cuts = {
    "video1.mp4": [(47, 59), (62, 68), (149, 192)],      # 0:47-0:59, 1:02-1:08, 2:29-3:12
    "video2.mp4": [(26, 31), (49, 57), (192, 200)],      # 0:26-0:31, 0:49-0:57, 3:12-3:20
    "video3.mp4": [(50, 58), (60, 67), (70, 75)],        # 0:50-0:58, 1:00-1:07, 1:10-1:15
    "video4.webm": [(3, 17), (30, 39)],                  # 0:03-0:17, 0:30-0:39
    "video5.webm": [(8, 21), (24, 29), (44, 50)]         # 0:08-0:21, 0:24-0:29, 0:44-0:50
}

# Process each video
for filename, intervals in cuts.items():
    input_path = os.path.join(input_dir, filename)
    clip = VideoFileClip(input_path)

    for i, (start, end) in enumerate(intervals, start=1):
        # Output filename will include the original video name + cut number
        output_path = os.path.join(output_dir, f"{filename}_cut{i}.mp4")

        # Extract subclip for the given time interval
        subclip = clip.subclip(start, end)

        # Export as MP4 with a standard codec (H.264) and audio included
        subclip.write_videofile(output_path, codec="libx264", audio=True)
        print(f"✅ Saved: {output_path}")

    clip.close()
