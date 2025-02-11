import pandas as pd

# Load the CSV file
input_file = "input_files/simulation_results.csv"
output_file = "input_files/filtered_results.csv"

# Read the CSV
df = pd.read_csv(input_file)

# Drop the QC column
df = df.drop(columns=["QC"])

# Filter rows where SF = 0.0
df_filtered = df[df["SF"] == 0.0]

# Drop the SF column after filtering
df_filtered = df_filtered.drop(columns=["SF"])

# Save the new CSV
df_filtered.to_csv(output_file, index=False)

print(f"Filtered CSV saved to: {output_file}")
