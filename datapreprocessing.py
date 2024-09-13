import pandas as pd

# Load CSV file
csv_path = "./data/B_images.csv"
data = pd.read_csv(csv_path)

# Normalize pixel values (0-255 to 0-1)
data = data / 255.0

# Add labels (adjust for each sign)
data['label'] = 'B'  # Change label based on the sign

# Save labeled data
labeled_csv_path = "./data/B_images_labeled.csv"
data.to_csv(labeled_csv_path, index=False)
print(f"Labeled data saved to {labeled_csv_path}")
