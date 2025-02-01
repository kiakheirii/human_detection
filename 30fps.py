import cv2
import os

# Video file path and output folder
video_path = r"D:\project\Part-A.MP4"
output_folder = '30fps_frames'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError("Error: Could not open the video file. Please check the path.")


# Get video FPS
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps is None:
    raise ValueError("Error: Could not retrieve FPS from the video.")

# Calculate the interval for extracting frames (every 1/30 seconds)
frame_interval = int(fps / 30) if fps >= 30 else 1

print(f"Video FPS: {fps}")
print(f"Frame Interval: {frame_interval}")

frame_count = 0
saved_frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if there are no more frames

    # Save every Nth frame based on the frame_interval
    if frame_count % frame_interval == 0:
        frame_filename = os.path.join(output_folder, f'frame_{saved_frame_count:04d}.jpg')
        cv2.imwrite(frame_filename, frame)
        saved_frame_count += 1

    frame_count += 1

cap.release()
print(f"Extracted {saved_frame_count} frames at 30 FPS.")
