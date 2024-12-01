import os
import joblib
import numpy as np
import logging

# Dynamically generate the log file name based on the script name
LOG_FILE = f"{os.path.splitext(os.path.basename(__file__))[0]}.log"

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    filemode='w'
)

# Model and scaler identification
MODEL_ID = '09'

# Specify the input and output folders
INPUT_FOLDER = 'input_files'
OUTPUT_FOLDER = 'output_files'

# Paths to the optimized model and scaler files
MODEL_PATH = os.path.join(OUTPUT_FOLDER, f'optimized_multi_model-{MODEL_ID}.joblib')
SCALER_PATH = os.path.join(INPUT_FOLDER, 'scaler2_x.joblib')

# Load the optimized model
try:
    multi_rf_model = joblib.load(MODEL_PATH)
    logging.info("Optimized Multi-output Random Forest model loaded successfully.")
except FileNotFoundError as e:
    logging.error(f"Model file not found: {MODEL_PATH}")
    raise e

# Load the input scaler
try:
    input_scaler = joblib.load(SCALER_PATH)
    logging.info("Input scaler loaded successfully.")
except FileNotFoundError as e:
    logging.error(f"Scaler file not found: {SCALER_PATH}")
    raise e

# Function to preprocess input data and make predictions
def predict_with_model(input_data):
    """
    Preprocess the input data using the scaler and predict the binary labels
    using the loaded model.
    
    Args:
        input_data (np.ndarray): A NumPy array containing the raw input data. Each row is a data point.
    
    Returns:
        dict: A dictionary containing probabilities and predicted classes.
    """
    try:
        # Scale the input data
        scaled_input_data = input_scaler.transform(input_data)
        
        # Predict probabilities
        predicted_probabilities = np.column_stack(
            [estimator.predict_proba(scaled_input_data)[:, 1] for estimator in multi_rf_model.estimators_]
        )
        
        # Predict binary classes
        predicted_classes = multi_rf_model.predict(scaled_input_data)
        
        # Log and return results
        logging.info(f"Input data shape: {input_data.shape}")
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
    # feedNH3 feedH2S QN1 QN2 QC SF
    example_input_data = np.array([        
        [0.003, 0.007, 500000.0, 800000.0, 3.0, 0.1]   # Example raw data point 1
        [0.005, 0.004, 600000.0, 900000.0, 3.0, 0.5]   # Example raw data point 1
    ])
    
    try:
        results = predict_with_model(example_input_data)
        
        # Print the results
        print("Predicted Probabilities:")
        print(results["predicted_probabilities"])
        
        print("\nPredicted Classes:")
        print(results["predicted_classes"])
        
    except Exception as e:
        print(f"An error occurred: {e}")
