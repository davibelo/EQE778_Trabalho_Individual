import os
import joblib
import numpy as np
import pandas as pd
import logging

# Dynamically generate the log file name based on the script name
LOG_FILE = f"{os.path.splitext(os.path.basename(__file__))[0]}.log"

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

# Model and scaler identification
MODEL_ID = '10'

# Specify the input and output folders
INPUT_FOLDER = 'input_files'
OUTPUT_FOLDER = 'output_files'

# Paths to the optimized model and scaler files
# Paths to the optimized model and scaler files
SCALER_FILE = 'scaler2_x.joblib'
MODELS_FILE = 'opt_model-{model_id}-output-{output_number}.joblib'

models = [
    os.path.join(OUTPUT_FOLDER, MODELS_FILE.format(model_id=MODEL_ID, output_number=i))
    for i in range(2)
]

rf_models = [joblib.load(model) for model in models]
logging.info("Optimized Models loaded successfully.")

input_scaler = joblib.load(os.path.join(INPUT_FOLDER, SCALER_FILE))
logging.info("Input scaler loaded successfully.")

# Function to preprocess input data and make predictions
def predict(input_data, input_scaler, models):
    try:
        # Scale the input data
        scaled_input_data = input_scaler.transform(input_data)

        # Initialize lists to store results
        predicted_probabilities = []
        predicted_classes = []

        # Loop through each model and make predictions
        for model in models:
            # Predict probabilities for the positive class (index 1)
            probabilities = model.predict_proba(scaled_input_data)[:, 1]
            predicted_probabilities.append(probabilities)

            # Predict binary classes
            classes = model.predict(scaled_input_data)
            predicted_classes.append(classes)

        # Convert results to numpy arrays
        predicted_probabilities = np.column_stack(predicted_probabilities)
        predicted_classes = np.column_stack(predicted_classes)

        # Log and return results
        logging.info(f"Input data shape: {input_data.shape}")
        logging.info(f"Predicted probabilities shape: {predicted_probabilities.shape}")
        logging.info(f"Predicted classes shape: {predicted_classes.shape}")
        logging.info("Prediction completed successfully.")

        return {
            "predicted_probabilities": predicted_probabilities,
            "predicted_classes": predicted_classes
        }
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise e

        
# Example usage
if __name__ == "__main__":
    # Define the column names
    column_names = ["feedNH3", "feedH2S", "QN1", "QN2", "SF"]

    # Example input data as a NumPy array
    example_input_data = np.array(
        [[0.004, 0.005, 600000.0, 700000.0, 1]]
    )

    # Convert the NumPy array to a pandas DataFrame with column names
    example_input_df = pd.DataFrame(example_input_data, columns=column_names)

    try:
        # Call the predict function with the example data
        results = predict(example_input_df, input_scaler, rf_models)
        
        # Print the results
        print("Predicted Probabilities:")
        print(results["predicted_probabilities"])
        
        print("\nPredicted Classes:")
        print(results["predicted_classes"])
        
    except Exception as e:
        print(f"An error occurred: {e}")
