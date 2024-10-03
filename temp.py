import os
import subprocess

def frames_to_video(input_frames_dir, output_video_path, frame_rate=30, image_format='png'):
    """
    Converts a sequence of image frames to a video using ffmpeg.

    :param input_frames_dir: Directory containing the image frames.
    :param output_video_path: Path to save the output video.
    :param frame_rate: Frame rate of the output video.
    :param image_format: Format of the image files (e.g., 'png', 'jpg').
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_video_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ffmpeg command to convert frames to video
    command = [
        'ffmpeg', '-framerate', str(frame_rate),
        '-i', f'{input_frames_dir}/%05d.{image_format}',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', output_video_path
    ]

    try:
        # Run the ffmpeg command
        subprocess.run(command, check=True)
        print(f"Video saved to {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during video conversion: {e}")

# Example usage
if __name__ == "__main__":
    frames_directory = 'output/dynerf/flame_salmon_orignal_exp2/test/ours_12000/gt'  # Directory containing image frames
    output_video = 'output/dynerf/flame_salmon_orignal_exp2/test/ours_12000/gt.mp4'    # Output video path
    frames_to_video(frames_directory, output_video, frame_rate=30, image_format='png')
