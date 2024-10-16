import pandas as pd

# List of CSV file paths
csv_files = ['train.csv', 'val.csv', 'test.csv']

# Read each CSV file into a DataFrame
dfs = [pd.read_csv(file) for file in csv_files]

# Concatenate all DataFrames
combined_df = pd.concat(dfs, ignore_index=True)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv('combined_file.csv', index=False)

print("CSV files have been merged into 'combined_file.csv'")

df = pd.read_csv('combined_file.csv')

# Shuffle the DataFrame rows
df_shuffled = df.sample(frac=1).reset_index(drop=True)

# Save the shuffled DataFrame back to a CSV file
df_shuffled.to_csv('shuffled_file.csv', index=False)

print("The rows have been shuffled and saved to 'shuffled_file.csv'")