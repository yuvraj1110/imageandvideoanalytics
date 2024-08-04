# -*- coding: utf-8 -*-
"""
                CSE-4076 Lab Assignment 2
 21MIA1102 
 yuvraj
"""

# Lab Task 1: Setup and Basic Extraction
import ffmpeg
import subprocess
import os

# Define the input video file path and output directory
input_video = r'C:\Users\KIIT\Downloads\3066446-uhd_4096_2160_24fps.mp4'
output_dir = r'C:\Users\KIIT\.spyder-py3\lab assignments'


# Define the FFmpeg command to extract frames
command = [
    'ffmpeg', 
    '-i', input_video,        # Input file
    '-vf', 'fps=1',           # Extract 1 frame per second
    os.path.join(output_dir, 'frame_%04d.png')  # Output file pattern
]
# Execute the command
subprocess.run(command)

print(f'Frames have been extracted to {output_dir}')



#TASK-2  Frame Type Analysis

import matplotlib.pyplot as plt

# Define the input video file path
input_video = r'C:\Users\KIIT\Downloads\3066446-uhd_4096_2160_24fps.mp4'

# FFmpeg command to count frame types
command = [
    'ffprobe', 
    '-v', 'error', 
    '-select_streams', 'v:0', 
    '-show_entries', 'frame=pict_type', 
    '-of', 'csv'
]

# Run the command and get the output
result = subprocess.run(command + [input_video], stdout=subprocess.PIPE, text=True)

# Parse the output to count frame types
frame_counts = {'I': 0, 'P': 0, 'B': 0}
for line in result.stdout.splitlines():
    if 'frame,' in line:
        frame_type = line.split(',')[1]
        if frame_type in frame_counts:
            frame_counts[frame_type] += 1

# Calculate total frames and percentages
total_frames = sum(frame_counts.values())
percentages = {k: (v / total_frames) * 100 for k, v in frame_counts.items()}

# Plot the distribution
plt.figure(figsize=(10, 6))
plt.bar(frame_counts.keys(), frame_counts.values(), color=['blue', 'orange', 'green'])
plt.xlabel('Frame Type')
plt.ylabel('Count')
plt.title('Distribution of Frame Types')
plt.show()

# Plot percentages
plt.figure(figsize=(10, 6))
plt.pie(percentages.values(), labels=percentages.keys(), autopct='%1.1f%%', colors=['blue', 'orange', 'green'])
plt.title('Percentage Distribution of Frame Types')
plt.show()

#TASK 3 Visualizing Frames
import cv2

# Create directories for frame types
frame_dirs = {'I': 'I_frames', 'P': 'P_frames', 'B': 'B_frames'}
for d in frame_dirs.values():
    if not os.path.exists(d):
        os.makedirs(d)

# Extract frames using FFmpeg
ffmpeg_command = 'ffmpeg -i {} -vf "select=eq(pict_type\\,I)" -vsync vfr {}/frame_%04d.png'
for frame_type, directory in frame_dirs.items():
    os.system(ffmpeg_command.format(input_video, directory))

# Display frames using OpenCV
for frame_type, directory in frame_dirs.items():
    frames = os.listdir(directory)
    for frame in frames:
        img = cv2.imread(os.path.join(directory, frame))
        cv2.imshow(f'{frame_type} frame', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

#Task 4: Frame Compression Analysis



# Calculate frame sizes
frame_sizes = {frame_type: [] for frame_type in frame_dirs.keys()}
for frame_type, directory in frame_dirs.items():
    for frame in os.listdir(directory):
        frame_path = os.path.join(directory, frame)
        frame_sizes[frame_type].append(os.path.getsize(frame_path))

# Calculate average sizes
average_sizes = {k: sum(v)/len(v) if v else 0 for k, v in frame_sizes.items()}
print("Average frame sizes (in bytes):\n", average_sizes)

# Compression efficiency discussion
print("I frames are intra-coded frames, meaning they are coded without reference to other frames. Therefore, they are typically larger in size.\n")
print("P frames are predicted frames, which use data from previous frames to compress more efficiently.\n")
print("B frames are bi-directional predicted frames, using data from both previous and future frames, making them usually the smallest.\n")


# Task 5: Advanced Frame Extraction


import glob

# Define output video parameters
output_video_path = 'I_frames_video.mp4'
fps = 1  # frames per second
frame_files = sorted(glob.glob('I_frames/frame_*.png'))
frame_height, frame_width, _ = cv2.imread(frame_files[0]).shape

# Create a video writer object
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Write frames to the video
for frame_file in frame_files:
    img = cv2.imread(frame_file)
    out.write(img)

out.release()
print(f'Video created at {output_video_path}')
