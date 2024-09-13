import pandas as pd

# File paths for the individual CSVs
file_A = './data/A_images_labeled.csv'
file_B = './data/B_images_labeled.csv'

# Output file path for the combined CSV
output_file = './data/combined_images_labeled.csv'

# Read the individual CSV files into DataFrames
df_A = pd.read_csv(file_A)
df_B = pd.read_csv(file_B)

# Combine the two DataFrames
combined_df = pd.concat([df_A, df_B], ignore_index=True)

# Save the combined DataFrame to a new CSV
combined_df.to_csv(output_file, index=False)

print(f"Combined CSV saved as {output_file}")
