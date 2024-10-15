import pandas as pd

# Load the CSV file
df = pd.read_csv('rainfall.csv')

# Perform linear interpolation to fill missing values in 'Daily Storage'
df['Actual Rainfall Value'] = df['Actual Rainfall Value'].interpolate(method='linear')

# Save the dataframe with filled values to a new CSV file
df.to_csv('rainfall.csv', index=False)
