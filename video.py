#!/usr/bin/env python3
"""
Create a side-by-side movie from PNG files in two folders.
"""

import os
import glob
import re
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def extract_number_from_filename(filename):
    """Extract number from filename for sorting."""
    match = re.search(r'(\d+)', os.path.basename(filename))
    return int(match.group(1)) if match else 0

def create_side_by_side_movie(folder1, folder2, output_video, fps=10):
    """
    Create side-by-side movie from PNG files in two folders.
    
    Parameters:
    -----------
    folder1 : str
        Path to first folder (left side)
    folder2 : str
        Path to second folder (right side)
    output_video : str
        Output video filename
    fps : int
        Frames per second
    """
    
    # Get all PNG files from both folders
    png_files1 = sorted(glob.glob(os.path.join(folder1, "*.png")), key=extract_number_from_filename)
    png_files2 = sorted(glob.glob(os.path.join(folder2, "*.png")), key=extract_number_from_filename)
    
    print(f"Found {len(png_files1)} files in {folder1}")
    print(f"Found {len(png_files2)} files in {folder2}")
    
    if not png_files1 or not png_files2:
        print("Error: No PNG files found in one or both folders!")
        return
    
    # Take minimum number of files to avoid index errors
    num_frames = min(len(png_files1), len(png_files2))
    print(f"Creating movie with {num_frames} frames")
    
    # Read first images to get dimensions
    img1 = cv2.imread(png_files1[0])
    img2 = cv2.imread(png_files2[0])
    
    if img1 is None or img2 is None:
        print("Error: Could not read first images!")
        return
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Make both images the same height by resizing
    target_height = min(h1, h2)
    new_w1 = int(w1 * target_height / h1)
    new_w2 = int(w2 * target_height / h2)
    
    # Total width for side-by-side video
    total_width = new_w1 + new_w2
    
    print(f"Output video dimensions: {total_width} x {target_height}")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (total_width, target_height))
    
    # Create frames
    for i in range(num_frames):
        # Read images
        img1 = cv2.imread(png_files1[i])
        img2 = cv2.imread(png_files2[i])
        
        if img1 is None or img2 is None:
            print(f"Warning: Could not read frame {i}")
            continue
        
        # Resize images to target height
        img1_resized = cv2.resize(img1, (new_w1, target_height))
        img2_resized = cv2.resize(img2, (new_w2, target_height))
        
        # Concatenate horizontally
        combined_frame = np.hstack((img1_resized, img2_resized))
        
        # Add labels to the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3
        thickness = 8
        color = (255, 255, 255) # White
        bg_color = (0, 0, 0)     # Black
        
        # Label 1: Full Metric (Left)
        text1 = "Full Metric (No Reduction)"
        text1_size = cv2.getTextSize(text1, font, font_scale, thickness)[0]
        text1_x = (new_w1 - text1_size[0]) // 2
        text1_y = 100
        
        # Draw background rectangle for text visibility
        cv2.rectangle(combined_frame, (text1_x - 10, text1_y - text1_size[1] - 10), 
                      (text1_x + text1_size[0] + 10, text1_y + 10), bg_color, -1)
        cv2.putText(combined_frame, text1, (text1_x, text1_y), font, font_scale, color, thickness, cv2.LINE_AA)
        
        # Label 2: Reduced Metric (Right)
        text2 = "Metric Reduced to FD"
        text2_size = cv2.getTextSize(text2, font, font_scale, thickness)[0]
        text2_x = new_w1 + (new_w2 - text2_size[0]) // 2
        text2_y = 100
        
        # Draw background rectangle for text visibility
        cv2.rectangle(combined_frame, (text2_x - 10, text2_y - text2_size[1] - 10), 
                      (text2_x + text2_size[0] + 10, text2_y + 10), bg_color, -1)
        cv2.putText(combined_frame, text2, (text2_x, text2_y), font, font_scale, color, thickness, cv2.LINE_AA)

        # Write frame
        out.write(combined_frame)
        
        if i % 10 == 0:
            print(f"Processed frame {i+1}/{num_frames}")
    
    # Release video writer
    out.release()
    print(f"Movie saved as: {output_video}")

def create_single_folder_movie(folder, output_video, fps=10):
    """
    Create a movie from PNG files in a single folder.
    """
    # Get all PNG files
    png_files = sorted(glob.glob(os.path.join(folder, "*.png")), key=extract_number_from_filename)
    
    print(f"Found {len(png_files)} files in {folder}")
    
    if not png_files:
        print(f"Error: No PNG files found in {folder}!")
        return
    
    num_frames = len(png_files)
    print(f"Creating movie with {num_frames} frames")
    
    # Read first image to get dimensions
    img = cv2.imread(png_files[0])
    if img is None:
        print("Error: Could not read first image!")
        return
    
    height, width = img.shape[:2]
    print(f"Output video dimensions: {width} x {height}")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Create frames
    for i in range(num_frames):
        img = cv2.imread(png_files[i])
        if img is None:
            print(f"Warning: Could not read frame {i}")
            continue
        
        # Write frame
        out.write(img)
        
        if i % 10 == 0:
            print(f"Processed frame {i+1}/{num_frames}")
    
    # Release video writer
    out.release()
    print(f"Movie saved as: {output_video}")

def combine_videos_side_by_side(video_path1, video_path2, output_video, fps=10):
    """
    Combine two existing videos side-by-side with labels.
    """
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)
    
    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open one or both videos!")
        return
        
    w1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    h1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    h2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # We'll normalize to a reasonable height (e.g., 1440p)
    target_height = 1440
    new_w1 = int(w1 * (target_height / h1))
    new_w2 = int(w2 * (target_height / h2))
    
    # Use avc1 (H.264) codec if possible, otherwise fall back
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (new_w1 + new_w2, target_height))
    
    frame_idx = 0
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
            
        # Resize both
        frame1 = cv2.resize(frame1, (new_w1, target_height))
        frame2 = cv2.resize(frame2, (new_w2, target_height))
        
        # Combine
        combined = np.hstack((frame1, frame2))
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        color = (255, 255, 255)
        bg_color = (0, 0, 0)
        
        # Label 1: Left
        text1 = "Full Metric (No Reduction)"
        t1_size = cv2.getTextSize(text1, font, font_scale, thickness)[0]
        t1_x = (new_w1 - t1_size[0]) // 2
        cv2.rectangle(combined, (t1_x - 10, 50 - t1_size[1] - 10), (t1_x + t1_size[0] + 10, 50 + 10), bg_color, -1)
        cv2.putText(combined, text1, (t1_x, 50), font, font_scale, color, thickness, cv2.LINE_AA)
        
        # Label 2: Right
        text2 = "Metric Reduced to EPN"
        t2_size = cv2.getTextSize(text2, font, font_scale, thickness)[0]
        t2_x = new_w1 + (new_w2 - t2_size[0]) // 2
        cv2.rectangle(combined, (t2_x - 10, 50 - t2_size[1] - 10), (t2_x + t2_size[0] + 10, 50 + 10), bg_color, -1)
        cv2.putText(combined, text2, (t2_x, 50), font, font_scale, color, thickness, cv2.LINE_AA)
        
        out.write(combined)
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Combined {frame_idx} frames...")
            
    cap1.release()
    cap2.release()
    out.release()
    print(f"Combined video saved as: {output_video}")

def create_side_by_side_movie_matplotlib(folder1, folder2, output_video, fps=10):
    """
    Alternative version using matplotlib for better control over layout.
    """
    import matplotlib.animation as animation
    
    # Get PNG files
    png_files1 = sorted(glob.glob(os.path.join(folder1, "*.png")), key=extract_number_from_filename)
    png_files2 = sorted(glob.glob(os.path.join(folder2, "*.png")), key=extract_number_from_filename)
    
    num_frames = min(len(png_files1), len(png_files2))
    print(f"Creating matplotlib movie with {num_frames} frames")
    
    # Set up the figure and axes
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    ax1.axis('off')
    ax2.axis('off')
    ax1.set_title('Full Metric (No Reduction)', fontsize=14, pad=20)
    ax2.set_title('Metric Reduced to EPN', fontsize=14, pad=20)
    
    def animate(frame):
        ax1.clear()
        ax2.clear()
        ax1.axis('off')
        ax2.axis('off')
        ax1.set_title(f'Full Metric (No Reduction) - Frame {frame+1}', fontsize=14, pad=20)
        ax2.set_title(f'Metric Reduced to EPN - Frame {frame+1}', fontsize=14, pad=20)
        
        # Load and display images
        img1 = plt.imread(png_files1[frame])
        img2 = plt.imread(png_files2[frame])
        
        ax1.imshow(img1)
        ax2.imshow(img2)
        
        return []
    
    # Create animation
    ani = animation.FuncAnimation(fig, animate, frames=num_frames, 
                                  interval=1000//fps, blit=False, repeat=True)
    
    # Save as MP4
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Python'), bitrate=1800)
    ani.save(output_video, writer=writer)
    
    plt.close(fig)
    print(f"Matplotlib movie saved as: {output_video}")

def main():
    """Main function to create the movie."""
    
    # Path to existing movies
    video1 = "no_reduction.mp4"        # Left side
    video2 = "elastic_reduction.mp4"   # Right side
    output_video = "poincare_comparison_combined.mp4"
    fps = 10
    
    if os.path.exists(video1) and os.path.exists(video2):
        print(f"Using existing videos to create comparison: {video1} and {video2}")
        combine_videos_side_by_side(video1, video2, output_video, fps)
    else:
        print("Existing movies not found, falling back to frame-based creation...")
        # Configuration for side-by-side comparison (frame folders)
        folder1 = "./figures_no_reduction"  # Left side (No Reduction)
        folder2 = "./figures"              # Right side (Elastic Reduction)
        output_video_fb = "poincare_comparison_side_by_side.mp4"
        
        # Check if folders exist
        if not os.path.exists(folder1) or not os.path.exists(folder2):
            print(f"Error: Folders '{folder1}' or '{folder2}' do not exist!")
            return
            
        try:
            create_side_by_side_movie(folder1, folder2, output_video_fb, fps)
        except Exception as e:
            print(f"OpenCV frame method failed: {e}")
            create_side_by_side_movie_matplotlib(folder1, folder2, output_video_fb.replace('.mp4', '_matplotlib.mp4'), fps)

if __name__ == "__main__":
    main()
