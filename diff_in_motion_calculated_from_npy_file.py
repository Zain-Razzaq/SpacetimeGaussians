import numpy as np
import os
import torch
import json
import cv2

def calculate_l1_loss(mag1, mag2):
    """Calculates the L1 loss between two magnitude arrays."""
    return np.mean(np.abs(mag1 - mag2))

def calculate_cosine_similarity_loss(angle1, angle2):
    """Calculates cosine similarity loss between two angle arrays."""
    cos_sim = np.cos(angle1 - angle2)
    return 1 - np.mean(cos_sim)  # 1 - mean cosine similarity gives loss


def process_and_save_images(set_name, flow, images_folder, output_folder):
    """Process images and save with motion arrows for a given set."""
    image_files = sorted([f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    image_files = sorted(image_files, key=lambda x: int(x.split('.')[0]))  # Sort the images based on numeric filenames
    # print image file names
    print(image_files)
    
    # Limit the number of images to 50, if more are present
    if len(image_files) > 50:
        image_files = image_files[:50]
    
    for idx in range(len(image_files)-1):
        image_file = image_files[idx+1]
        image_path = os.path.join(images_folder, image_file)
        image = cv2.imread(image_path)
        
        # Since flow shape is (78, 1014, 1352), corresponding to 39 frames (u and v for each frame)
        # Each frame has u at 2*idx and v at 2*idx + 1
        flow_u = flow[2 * idx]      # Extract u component
        flow_v = flow[2 * idx + 1]  # Extract v component

        # apply threshold to remove very small flow values
        threshold = 0.1
        flow_u[flow_u < threshold] = 0
        flow_v[flow_v < threshold] = 0
        
        output_image_path = os.path.join(output_folder, f"{set_name}_motion_image_{image_file}")
        
        # Call the function to save the image with arrows showing motion vectors
        for y in range(0, image.shape[0], 10):
            for x in range(0, image.shape[1], 10):
                if abs(flow_u[y, x]) > threshold or abs(flow_v[y, x]) > threshold:
                    cv2.arrowedLine(image, (x, y), (int(x + flow_u[y, x]), int(y + flow_v[y, x])), (0, 255, 0), 2)

        # Save the image
        cv2.imwrite(output_image_path, image)


def main():
    # Load magnitudes and angles from npy files for both sets
    set1_mag_path = 'data/OF_experiment_data/cam09_gt/output/magnitudes.npy'
    set2_mag_path = 'data/OF_experiment_data/train_renders_cam09/output/magnitudes.npy'
    images_folder_set1 = 'data/OF_experiment_data/cam09_gt'
    images_folder_set2 = 'data/OF_experiment_data/train_renders_cam09'
    flow1 = 'data/OF_experiment_data/cam09_gt/output/flow.npy'
    flow2 = 'data/OF_experiment_data/train_renders_cam09/output/flow.npy'
    output_folder = 'data/OF_experiment_data/output'


    output_set1 = os.path.join(output_folder, "set1_images")
    output_set2 = os.path.join(output_folder, "set2_images")
    if not os.path.exists(output_set1):
        os.makedirs(output_set1)
    if not os.path.exists(output_set2):
        os.makedirs(output_set2)

    magnitudes1 = np.load(set1_mag_path)
    magnitudes2 = np.load(set2_mag_path)
    flow1 = np.load(flow1)
    flow2 = np.load(flow2)

    print(f"Shape of magnitudes1: {magnitudes1.shape}")
    print(f"Shape of magnitudes2: {magnitudes2.shape}")
    print(f"Shape of flow1: {flow1.shape}")
    print(f"Shape of flow2: {flow2.shape}")

    # Initialize accumulators for loss calculations
    total_l1_loss = []
    total_cosine_loss = []
    N = len(magnitudes2)  # Assuming both sets have the same number of images
    
    # Calculate losses for all images and sum them
    for idx in range(N):
        # apply threshold to remove very small flow values
        threshold = 0.1
        flow1[2*idx][flow1[2*idx] < threshold] = 0
        flow1[2*idx + 1][flow1[2*idx + 1] < threshold] = 0
        flow2[2*idx][flow2[2*idx] < threshold] = 0
        flow2[2*idx + 1][flow2[2*idx + 1] < threshold] = 0

        # calculate magnitude from flow
        mag1 = np.sqrt(flow1[2*idx] ** 2 + flow1[2*idx + 1] ** 2)
        mag2 = np.sqrt(flow2[2*idx] ** 2 + flow2[2*idx + 1] ** 2)
        l1_loss = calculate_l1_loss(mag1, mag2)
        # calculate angle from flow
        angle1 = np.arctan2(flow1[2*idx + 1], flow1[2*idx])
        angle2 = np.arctan2(flow2[2*idx + 1], flow2[2*idx])

        # calculate cosine similarity loss
        cosine_loss = calculate_cosine_similarity_loss(angle1, angle2)
        
        total_l1_loss.append(l1_loss)
        total_cosine_loss.append(cosine_loss)

    # Average losses over all images
    avg_l1_loss = np.mean(total_l1_loss)
    avg_cosine_loss = np.mean(total_cosine_loss)

    # Save the  losses and mean min and max losses to a JSON file
    with open(os.path.join(output_folder, 'losses.txt'), 'w') as f:
        for idx, (l1_loss, cosine_loss) in enumerate(zip(total_l1_loss, total_cosine_loss)):
            f.write(f"Image {idx + 1} - L1 Loss: {l1_loss:.4f}, Cosine Loss: {cosine_loss:.4f}\n")
        f.write(f"\nAverage L1 Loss: {avg_l1_loss:.4f}\n")
        f.write(f"Average Cosine Similarity Loss: {avg_cosine_loss:.4f}\n")

    process_and_save_images("set1", flow1, images_folder_set1, output_set1)
    process_and_save_images("set2", flow2, images_folder_set2, output_set2)
    
    print(f"Results saved in {output_folder}")

if __name__ == "__main__":
    main()

