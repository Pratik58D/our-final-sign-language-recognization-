import os
import cv2
import numpy as np
import pandas as pd

# Directory of images
folder = "./data/B"  # Change based on the sign

# Verify the directory exists
if not os.path.exists(folder):
    print(f"The folder '{folder}' does not exist.")
    exit()

# List of image files
image_files = [f for f in os.listdir(folder) if f.endswith('.jpg')]

# Check if there are images in the folder
if not image_files:
    print(f"No images found in the folder '{folder}'.")
    exit()

# List to store image data
image_data = []

# Process each image
for image_file in image_files:
    img_path = os.path.join(folder, image_file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
    if img is None:
        print(f"Error reading image {img_path}. Skipping.")
        continue

    img = cv2.resize(img, (300, 300))  # Resize to 300x300
    img_flattened = img.flatten()  # Flatten image into a 1D array
    image_data.append(img_flattened)

# Convert list to DataFrame
df = pd.DataFrame(image_data)

# Save DataFrame to CSV
csv_path = "./data/B_images.csv"  # Change based on the sign
df.to_csv(csv_path, index=False)
print(f"CSV file saved at {csv_path}")
