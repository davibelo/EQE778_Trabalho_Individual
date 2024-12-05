import os
import joblib
import logging
import win32com.client as win32
import numpy as np
import pandas as pd
from scipy.optimize import minimize

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
SCALER_FILE = 'scaler2_x.joblib'
MODELS_FILE = 'opt_model-{model_id}-output-{output_number}.joblib'

# Paths to columns names
COLUMNS_X_FILE = 'df2_scaled_x_columns.joblib'
COLUMNS_Y_FILE = 'df2_bin_y_columns.joblib'

# Load the optimized models and input scaler
models_paths = [
    os.path.join(OUTPUT_FOLDER, MODELS_FILE.format(model_id=MODEL_ID, output_number=i))
    for i in range(2)
]

models = [joblib.load(model) for model in models_paths]
logging.info("Optimized Models loaded successfully.")

input_scaler = joblib.load(os.path.join(INPUT_FOLDER, SCALER_FILE))
logging.info("Input scaler loaded successfully.")

# Load columns names
columns_x = joblib.load(os.path.join(INPUT_FOLDER, COLUMNS_X_FILE))
columns_y = joblib.load(os.path.join(INPUT_FOLDER, COLUMNS_Y_FILE))

# Aspen Plus connection
ASPEN_FILE_FOLDER = 'UTAA_run'  # Specify the folder where the Aspen Plus file is located
ASPEN_FILE = 'UTAA_revK.bkp'  # File name
aspen_path = os.path.abspath(os.path.join(ASPEN_FILE_FOLDER, ASPEN_FILE))  # Build the full path

# print('Connecting to the Aspen Plus... Please wait ')
# Application = win32.Dispatch('Apwn.Document')  # Registered name of Aspen Plus
# print('Connected!')

# Application.InitFromArchive2(aspen_path)
# Application.visible = 0

# Function to preprocess input data and make predictions
def predict(input_data, input_scaler, models):
    try:
        # Scale the input data        
        scaled_input_data = input_scaler.transform([input_data])

        # Initialize lists to store results
        predicted_probabilities = []
        predicted_classes = []       

        # Loop through each model and make predictions
        for model in models:
            # Predict probabilities for the positive class (index 1)
            probability = model.predict_proba(scaled_input_data)[:, 1]
            predicted_probabilities.append(probability)

            # Predict binary class
            classification = model.predict(scaled_input_data)
            predicted_classes.append(classification)

        # Log and return results
        logging.info(f"Input data shape: {np.array(input_data).shape}")
        logging.info(f"Predicted probabilities: {predicted_probabilities}")
        logging.info(f"Predicted classes: {predicted_classes}")
        logging.info("Prediction completed successfully.")

        return {
            "predicted_probabilities": predicted_probabilities,
            "predicted_classes": predicted_classes
        }
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise e

# Objective function to minimize
def cost(x_opt_scaled):
    x = [x_opt_scaled[i] * opt_scale_factors[i] for i in range(len(x_opt_scaled))]
    total_cost = x[0] + x[1]
    x_values.append(x)  # Store non-scaled x
    objective_values.append(total_cost)  # Store objective function value
    logging.info(f"Total Cost: {total_cost}")
    return total_cost

# Constraint 1 (H2S PPM <= 0.2)
def constraint1(x_opt_scaled):
    x = [x_opt_scaled[i] * opt_scale_factors[i] for i in range(len(x_opt_scaled))]
    inputs = np.array([feedNH3, feedH2S, x[0], x[1], x[2]])
    results = predict(inputs, input_scaler, models)
    cH2S_prob = results["predicted_probabilities"][0]
    # Return the constraint value (positive if satisfied, negative if violated)
    return 0.5 - cH2S_prob

# Constraint 2 (NH3 PPM <= 15)
def constraint2(x_opt_scaled):
    x = [x_opt_scaled[i] * opt_scale_factors[i] for i in range(len(x_opt_scaled))]
    inputs = np.array([feedNH3, feedH2S, x[0], x[1], x[2]])    
    results = predict(inputs, input_scaler, models)
    cNH3_prob = results["predicted_probabilities"][1]
    # Return the constraint value (positive if satisfied, negative if violated)
    return 0.5 - cNH3_prob

# Lower bound constraint for QN1
def bound_QN1_lower(x_opt_scaled):
    QN1_lower = 450000
    return x_opt_scaled[0] - (QN1_lower / opt_scale_factors[0])

# Upper bound constraint for QN1
def bound_QN1_upper(x_opt_scaled):
    QN1_upper = 600000
    return (QN1_upper / opt_scale_factors[0]) - x_opt_scaled[0]

# Lower bound constraint for QN2
def bound_QN2_lower(x_opt_scaled):
    QN2_lower = 700000
    return x_opt_scaled[1] - (QN2_lower / opt_scale_factors[1])

# Upper bound constraint for QN2
def bound_QN2_upper(x_opt_scaled):
    QN2_upper = 1200000
    return (QN2_upper / opt_scale_factors[1]) - x_opt_scaled[1]

# Lower bound constraint for SF
def bound_SF_lower(x_opt_scaled):
    SF_lower = 0
    return x_opt_scaled[2] - (SF_lower / opt_scale_factors[2])

# Upper bound constraint for SF
def bound_SF_upper(x_opt_scaled):
    SF_upper = 1
    return (SF_upper / opt_scale_factors[2]) - x_opt_scaled[2]

# Fixed feed quality
feedNH3 = 0.004
feedH2S = 0.005

# Initial guess
x0 = [560000, 950000, 0.5]  # QN1, QN2, SF

# Scaling factors for optimization
opt_scale_factors = [1e5, 1e5, 0.1]  # Scaling for QN1, QN2, SF

# Initial guess (with optimization scaling)
x0_opt_scaled = [x0[i] / opt_scale_factors[i] for i in range(len(x0))]

# Define constraints as a list of dictionaries
constraints = [
    {'type': 'ineq', 'fun': constraint1},      # H2S constraint
    {'type': 'ineq', 'fun': constraint2},      # NH3 constraint
    {'type': 'ineq', 'fun': bound_QN1_lower},  # QN1 lower bound
    {'type': 'ineq', 'fun': bound_QN1_upper},  # QN1 upper bound
    {'type': 'ineq', 'fun': bound_QN2_lower},  # QN2 lower bound
    {'type': 'ineq', 'fun': bound_QN2_upper},  # QN2 upper bound
    {'type': 'ineq', 'fun': bound_SF_lower},   # SF lower bound
    {'type': 'ineq', 'fun': bound_SF_upper}    # SF upper bound
]

options = {
    'maxiter': 10000,
    'tol': 1e-2
}

# Lists to store results
x_values = []
objective_values = []

# Solving the optimization problem
result = minimize(cost, x0_opt_scaled, method='COBYLA', constraints=constraints, options=options)

# Rescale the results
opt_scaled = result.x
opt = [opt_scaled[i] * opt_scale_factors[i] for i in range(len(opt_scaled))]

# Format the output values
formatted_opt = f"[{opt[0]:.0f}, {opt[1]:.0f}, {opt[2]:.2f}]"

# Output results
cost_min = result.fun
num_function_evals = result.nfev
success = result.success
message = result.message
maxcv = result.maxcv  # Magnitude of constraint violation

logging.info(f'Optimal values: {formatted_opt}')
logging.info(f'Minimum cost: {cost_min}')
logging.info(f'Number of function evaluations: {num_function_evals}')
logging.info(f'Optimization success: {success}')
logging.info(f'Message: {message}')
logging.info(f'Maximum constraint violation (maxcv): {maxcv}')
