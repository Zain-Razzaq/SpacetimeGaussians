import cv2
import numpy as np
from argparse import ArgumentParser


# take argument from command line
parser = ArgumentParser()
parser.add_argument("-i", "--input", required=True, help="Path to the input video")
args = vars(parser.parse_args())

# read number of cameras
# number of cameras is equal to the folders coantained in the input folder named cam00, cam01, cam02, ...
# each folder contains the frames of the video captured by the camera
input_folder = args["input"]
cameras = []

# read all the folders in the input folder
import os
for folder in os.listdir(input_folder):
    # count the number of folders having the name starting with cam
    if folder.startswith("cam") and os.path.isdir(os.path.join(input_folder, folder)):
        cameras.append(folder)  

print(f"Number of cameras: {len(cameras)}")

# make a subprocesss to run another python script for each camera
import subprocess
for camera in cameras:
    # run another python script for each camera
    subprocess.run(["python", "./inference.py", "--name", "MemFlowNet", "--stage", "spring_only", "--restore_ckpt", "ckpts/MemFlowNet_spring.pth", "--seq_dir", os.path.join(input_folder, camera), "--vis_dir", os.path.join(input_folder, camera)])
    # wait for the subprocess to finish and then continue with the next camera
    print(f"Finished processing camera: {camera}")


# 
