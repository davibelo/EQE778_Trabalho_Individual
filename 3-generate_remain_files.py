import pandas as pd
import os

# File paths
initial_files = [
    "simulation_points_part_1_initial.csv",
    "simulation_points_part_2_initial.csv",
    "simulation_points_part_3_initial.csv",
    "simulation_points_part_4_initial.csv"
]
results_files = [
    "simulation_results_part_1.csv",
    "simulation_results_part_2.csv",
    "simulation_results_part_3.csv",
    "simulation_results_part_4.csv"
]

# Loop through each initial and results file
for i, (initial_file, results_file) in enumerate(zip(initial_files, results_files), start=1):
    # Load initial and results data
    initial_df = pd.read_csv(initial_file)
    results_df = pd.read_csv(results_file)

    # Identify columns in initial file that define a unique point
    initial_columns = initial_df.columns  # Use all columns from initial file
    results_columns = results_df.columns[:-2]  # Exclude last two columns in results file, which are the results

    # Merge dataframes to find common and non-common rows based on initial columns
    merged_df = pd.merge(initial_df, results_df[results_columns], on=list(initial_columns), how='left', indicator=True)
    
    # Filter out rows that are already simulated (present in results)
    remaining_df = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])

    # Shuffle the rows of the remaining dataframe
    remaining_df = remaining_df.sample(frac=1).reset_index(drop=True)

    # Define the output file path
    output_file = f"remain_sim_points_part_{i}.csv"
    
    # Check if the output file already exists and delete it if necessary
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"Existing file {output_file} deleted.")

    # Save shuffled remaining points to a new CSV file
    remaining_df.to_csv(output_file, index=False)
    print(f"Remaining points saved to {output_file}")
