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
ASPEN_FILE_FOLDER = 'UTAA_run'  # Specify the folder where the Aspen Plus file is located
ASPEN_FILE = 'UTAA_revK.bkp'  # File name
aspen_path = os.path.abspath(os.path.join(ASPEN_FILE_FOLDER, ASPEN_FILE))  # Build the full path

# print('Connecting to the Aspen Plus... Please wait ')
# Application = win32.Dispatch('Apwn.Document')  # Registered name of Aspen Plus
# print('Connected!')

# Application.InitFromArchive2(aspen_path)
# Application.visible = 0

# Function to preprocess input data and make predictions
def predict(fixed_inputs, optimization_inputs, input_scaler, models):
    try:
        # Ensure both inputs are arrays for compatibility
        fixed_inputs = np.array(fixed_inputs)
        optimization_inputs = np.array(optimization_inputs)

        # Combine fixed inputs and optimization inputs
        full_input = np.concatenate((fixed_inputs, optimization_inputs))

        # Scale the input data
        scaled_input_data = input_scaler.transform([full_input])

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
        logging.info(f"Input data shape: {np.array(full_input).shape}")
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
def cost(opt_inputs_scaled, fixed_inputs, input_scaler):
    fixed_inputs = np.array(fixed_inputs)
    opt_inputs_scaled = np.array(opt_inputs_scaled)

    full_input_scaled = np.concatenate((fixed_inputs, opt_inputs_scaled))
    full_input = input_scaler.inverse_transform([full_input_scaled])[0]
    total_cost = full_input[2] + full_input[3]
    x_values.append(full_input)  # Store non-scaled full input
    objective_values.append(total_cost)  # Store objective function value
    logging.info(f"Total Cost: {total_cost}")
    return total_cost

# Constraint 1 (H2S PPM <= 0.2. bin prob <= 0.5)
def constraint1(opt_inputs_scaled, fixed_inputs, input_scaler, models):
    fixed_inputs = np.array(fixed_inputs)
    opt_inputs_scaled = np.array(opt_inputs_scaled)
    full_input_scaled = np.concatenate((fixed_inputs, opt_inputs_scaled))
    full_input = input_scaler.inverse_transform([full_input_scaled])[0]
    results = predict(fixed_inputs, opt_inputs_scaled, input_scaler, models)
    cH2S_prob = results["predicted_probabilities"][0]
    return 0.5 - cH2S_prob

# Constraint 2 (NH3 PPM <= 15, bin prob <= 0.5)
def constraint2(opt_inputs_scaled, fixed_inputs, input_scaler, models):
    fixed_inputs = np.array(fixed_inputs)
    opt_inputs_scaled = np.array(opt_inputs_scaled)
    full_input_scaled = np.concatenate((fixed_inputs, opt_inputs_scaled))
    full_input = input_scaler.inverse_transform([full_input_scaled])[0]
    results = predict(fixed_inputs, opt_inputs_scaled, input_scaler, models)
    cNH3_prob = results["predicted_probabilities"][1]
    return 0.5 - cNH3_prob

# Bound constraints (already in scaled space)
def bound_QN1_lower(opt_inputs_scaled):
    return opt_inputs_scaled[0] - input_scaler.transform([[0, 0, 450000, 0, 0]])[0, 2]

def bound_QN1_upper(opt_inputs_scaled):
    return input_scaler.transform([[0, 0, 600000, 0, 0]])[0, 2] - opt_inputs_scaled[0]

def bound_QN2_lower(opt_inputs_scaled):
    return opt_inputs_scaled[1] - input_scaler.transform([[0, 0, 0, 700000, 0]])[0, 3]

def bound_QN2_upper(opt_inputs_scaled):
    return input_scaler.transform([[0, 0, 0, 1200000, 0]])[0, 3] - opt_inputs_scaled[1]

def bound_SF_lower(opt_inputs_scaled):
    return opt_inputs_scaled[2] - input_scaler.transform([[0, 0, 0, 0, 0]])[0, 4]

def bound_SF_upper(opt_inputs_scaled):
    return input_scaler.transform([[0, 0, 0, 0, 1]])[0, 4] - opt_inputs_scaled[2]

# Fixed inputs
fixed_inputs = input_scaler.transform([[0.005, 0.004, 0, 0, 0]])[0][:2]  # feedNH3, feedH2S (scaled)

# Initial guess for optimization variables (scaled)
x0 = [560000, 950000, 0.5]  # QN1, QN2, SF
x0_scaled = input_scaler.transform([[0.005, 0.004] + x0])[0][2:]
logging.info(f"Initial guess (scaled): {x0_scaled}")

# Define constraints as a list of dictionaries
constraints = [
    {'type': 'ineq', 'fun': lambda x: constraint1(x, fixed_inputs, input_scaler, models)},
    {'type': 'ineq', 'fun': lambda x: constraint2(x, fixed_inputs, input_scaler, models)},
    {'type': 'ineq', 'fun': lambda x: bound_QN1_lower(x)},
    {'type': 'ineq', 'fun': lambda x: bound_QN1_upper(x)},
    {'type': 'ineq', 'fun': lambda x: bound_QN2_lower(x)},
    {'type': 'ineq', 'fun': lambda x: bound_QN2_upper(x)},
    {'type': 'ineq', 'fun': lambda x: bound_SF_lower(x)},
    {'type': 'ineq', 'fun': lambda x: bound_SF_upper(x)}
]

options = {
    'maxiter': 10000,
    'tol': 1e-2
}

# Lists to store results
x_values = []
objective_values = []

# Solving the optimization problem
result = minimize(
    cost,
    x0_scaled,
    method='COBYLA',
    constraints=constraints,
    options=options,
    args=(fixed_inputs, input_scaler)
)

# Rescale the results
opt_scaled = result.x
opt_full_scaled = np.concatenate((fixed_inputs, opt_scaled))
opt = input_scaler.inverse_transform([opt_full_scaled])[0]

# Format the output values
formatted_opt = f"[{opt[0]:.4f}, {opt[1]:.4f}, {opt[2]:.0f}, {opt[3]:.0f}, {opt[4]:.2f}]"

# Output results
cost_min = result.fun
num_function_evals = result.nfev
success = result.success
message = result.message

logging.info(f'Optimal values: {formatted_opt}')
logging.info(f'Minimum cost: {cost_min}')
logging.info(f'Number of function evaluations: {num_function_evals}')
logging.info(f'Optimization success: {success}')
logging.info(f'Message: {message}')
