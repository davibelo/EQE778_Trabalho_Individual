import pandas as pd
import glob
import os

# Get list of all CSV files in the current directory that start with 'simulation_results_'
csv_files = glob.glob("simulation_results_*.csv")

# Initialize an empty list to hold each DataFrame
dataframes = []

# Loop through the list of files and read each one into a DataFrame
for file in csv_files:
    df = pd.read_csv(file)
    dataframes.append(df)

# Concatenate all DataFrames
if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)
    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv("simulation_results.csv", index=False)
    print("All files have been combined and saved to simulation_results.csv")
else:
    print("No files found to combine.")
