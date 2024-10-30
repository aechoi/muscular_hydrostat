import cv2
import os

# Set the directory and desired frame rate
frames_dir = "frames"
output_video = "simulation_video.mp4"
fps = 60  # Adjust as needed for smooth playback

# Get all frame file names and sort them
frame_files = sorted(
    [os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".png")]
)

# Get the size of the first frame to set video dimensions
first_frame = cv2.imread(frame_files[0])
height, width, layers = first_frame.shape
video = cv2.VideoWriter(
    output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
)

# Write frames to the video
for frame_file in frame_files:
    frame = cv2.imread(frame_file)
    video.write(frame)

video.release()
print("Video created successfully!")
