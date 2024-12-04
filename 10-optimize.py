import os
import joblib
import logging
import win32com.client as win32
import numpy as np
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
aspen_file = r"UTAA_revK.bkp"
aspen_path = os.path.abspath(aspen_file)

print('Connecting to the Aspen Plus... Please wait ')
Application = win32.Dispatch('Apwn.Document')  # Registered name of Aspen Plus
print('Connected!')

Application.InitFromArchive2(aspen_path)
Application.visible = 0

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
            # Predict probabilities for the positive class
            probabilities = model.predict_proba(scaled_input_data)[:, 1]
            predicted_probabilities.append(probabilities)

            # Predict binary class
            classes = model.predict(scaled_input_data)
            predicted_classes.append(classes)

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
def cost(x_scaled):
    x = input_scaler.inverse_transform([x_scaled])[0]
    total_cost = x[2] + x[3]
    x_values.append(x)  # Store non-scaled x
    objective_values.append(total_cost)  # Store objective function value
    logging.info(f"Total Cost: {total_cost}")
    return total_cost

# Constraint 1 (H2S PPM >= 0.2)
def constraint1(x_scaled):
    x = input_scaler.inverse_transform([x_scaled])[0]
    results = predict(x, input_scaler, models)
    cH2S_prob = results["predicted_probabilities"][0]
    return cH2S_prob - 0.6

# Constraint 2 (NH3 PPM >= 15)
def constraint2(x_scaled):
    x = input_scaler.inverse_transform([x_scaled])[0]
    results = predict(x, input_scaler, models)
    cNH3_prob = results["predicted_probabilities"][1]
    return cNH3_prob - 0.6

# Bound constraints (already in scaled space)
def bound_QN1_lower(x_scaled):
    return x_scaled[2] - input_scaler.transform([[0, 0, 450000, 0, 0]])[0, 2]

def bound_QN1_upper(x_scaled):
    return input_scaler.transform([[0, 0, 600000, 0, 0]])[0, 2] - x_scaled[2]

def bound_QN2_lower(x_scaled):
    return x_scaled[3] - input_scaler.transform([[0, 0, 0, 700000, 0]])[0, 3]

def bound_QN2_upper(x_scaled):
    return input_scaler.transform([[0, 0, 0, 1200000, 0]])[0, 3] - x_scaled[3]

def bound_SF_lower(x_scaled):
    return x_scaled[4] - input_scaler.transform([[0, 0, 0, 0, 0]])[0, 4]

def bound_SF_upper(x_scaled):
    return input_scaler.transform([[0, 0, 0, 0, 1]])[0, 4] - x_scaled[4]

# Initial guess
x0 = [0.005, 0.004, 560000, 950000, 0.5]  # feedNH3, feedH2S, QN1, QN2, SF
x0_scaled = input_scaler.transform([x0])[0]

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
result = minimize(cost, x0_scaled, method='COBYLA', constraints=constraints, options=options)

# Rescale the results
opt_scaled = result.x
opt = input_scaler.inverse_transform([opt_scaled])[0]

# Output results
cost_min = result.fun
num_function_evals = result.nfev
success = result.success
message = result.message
maxcv = result.maxcv  # Magnitude of constraint violation

logging.info(f'Optimal values: {opt}')
logging.info(f'Minimum cost: {cost_min}')
logging.info(f'Number of function evaluations: {num_function_evals}')
logging.info(f'Optimization success: {success}')
logging.info(f'Message: {message}')
logging.info(f'Maximum constraint violation (maxcv): {maxcv}')
