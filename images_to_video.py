import os
import subprocess

def images_to_video(image_folder, output_video, frame_rate=30):
    # Check if the image folder exists
    if not os.path.isdir(image_folder):
        raise FileNotFoundError(f"The folder {image_folder} does not exist.")
    
    # Prepare the ffmpeg command
    command = [
        'ffmpeg',
        '-framerate', str(frame_rate),
        '-i', os.path.join(image_folder, '%04d.png'),  # Assuming images are named in format 0001.png, 0002.png, etc.
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_video
    ]
    
    # Execute the ffmpeg command
    try:
        subprocess.run(command, check=True)
        print(f"Video saved as {output_video}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

# Example usage
image_folder = 'data/ball_slow/frames/cam00'
output_video = '0.mp4'
images_to_video(image_folder, output_video, frame_rate=30)
