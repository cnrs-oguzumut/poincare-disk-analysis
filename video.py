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
        
        # Write frame
        out.write(combined_frame)
        
        if i % 10 == 0:
            print(f"Processed frame {i+1}/{num_frames}")
    
    # Release video writer
    out.release()
    print(f"Movie saved as: {output_video}")

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
    ax1.set_title('Figures')
    ax2.set_title('Delaunay')
    
    def animate(frame):
        ax1.clear()
        ax2.clear()
        ax1.axis('off')
        ax2.axis('off')
        ax1.set_title(f'Figures - Frame {frame+1}')
        ax2.set_title(f'Delaunay - Frame {frame+1}')
        
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
    
    # Configuration
    folder1 = "./figures"      # Left side
    folder2 = "./delanuay"     # Right side (note: you wrote "delanuay" - is this correct?)
    output_video = "side_by_side_comparison.mp4"
    fps = 5  # Adjust as needed
    
    # Check if folders exist
    if not os.path.exists(folder1):
        print(f"Error: Folder '{folder1}' does not exist!")
        return
    
    if not os.path.exists(folder2):
        print(f"Error: Folder '{folder2}' does not exist!")
        return
    
    print("Creating side-by-side movie...")
    print(f"Left side: {folder1}")
    print(f"Right side: {folder2}")
    print(f"Output: {output_video}")
    print(f"FPS: {fps}")
    print("-" * 50)
    
    try:
        # Try OpenCV method first
        create_side_by_side_movie(folder1, folder2, output_video, fps)
    except Exception as e:
        print(f"OpenCV method failed: {e}")
        print("Trying matplotlib method...")
        try:
            create_side_by_side_movie_matplotlib(folder1, folder2, 
                                               output_video.replace('.mp4', '_matplotlib.mp4'), fps)
        except Exception as e2:
            print(f"Matplotlib method also failed: {e2}")
            print("Please install required packages: pip install opencv-python matplotlib")

if __name__ == "__main__":
    main()
