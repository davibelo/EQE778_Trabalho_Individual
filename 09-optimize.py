import os
import joblib
import numpy as np
import logging
import win32com.client as win32
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

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

aspen_file = r"UTAA_revK.bkp"
aspen_path = os.path.abspath(aspen_file)

print('Connecting to the Aspen Plus... Please wait ')
Application = win32.Dispatch('Apwn.Document')  # Registered name of Aspen Plus
print('Connected!')

Application.InitFromArchive2(aspen_Path)
Application.visible = 0

# Initial guess
x0 = [0.005, 0.004, 560000, 950000, 0.5] # feedNH3 feedH2S QN1 QN2 SF

# Scaling factors
scale_factors = [0.01, 0.01, 1e5, 1e5, 0.1]

# Lists to store non-scaled x values and corresponding objective function values
x_values = []
objective_values = []

# Function to preprocess input data and make predictions
def predict(input_data):
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
 
# Objective function to minimize (with scaling)
def cost(x_scaled):
    x = [x_scaled[0] * scale_factors[0], 
         x_scaled[1] * scale_factors[1], 
         x_scaled[2] * scale_factors[2], 
         x_scaled[3] * scale_factors[3],
         x_scaled[4] * scale_factors[4]]
    total_cost = x[2] + x[3]
    # Store the non-scaled x values and total cost
    x_values.append(x)  # Store non-scaled x
    objective_values.append(total_cost)  # Store objective function value
    logging.info(f"Total Cost: {total_cost}")
    return total_cost

# Constraint 1 (H2S PPM <= 0.2)
def constraint1(x_scaled):
    results = predict(x_scaled)
    cH2S_prob = results["predicted_probabilities"][0]
    return 0.9 - cH2S_prob  # >=0.9

# Constraint 2 (NH3 PPM <= 15)
def constraint2(x_scaled):
    results = predict(x_scaled)
    cNH3_prob = results["predicted_probabilities"][1]
    return 15 - cNH3_prob  # >=0.9

# Lower bound constraint for QN1
def bound_QN1_lower(x_scaled):
    QN1_lower = 450000
    return x_scaled[2] - (QN1_lower / scale_factors[0])

# Upper bound constraint for QN1
def bound_QN1_upper(x_scaled):
    QN1_upper = 600000
    return (QN1_upper / scale_factors[2]) - x_scaled[2]

# Lower bound constraint for QN2
def bound_QN2_lower(x_scaled):
    QN2_lower = 700000
    return x_scaled[3] - (QN2_lower / scale_factors[3])

# Upper bound constraint for QN2
def bound_QN2_upper(x_scaled):
    QN2_upper = 1200000
    return (QN2_upper / scale_factors[3]) - x_scaled[3]

# Lower bound constraint for SF
def bound_SF_lower(x_scaled):
    SF_lower = 0
    return x_scaled[4] - (SF_lower / scale_factors[4])

# Upper bound constraint for SF
def bound_SF_upper(x_scaled):
    SF_upper = 1
    return (SF_upper / scale_factors[4]) - x_scaled[4]

# Initial guess (with scaling)
x0_scaled = [x0[i] / scale_factors[i] for i in range(4)]

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

# Solving the optimization problem with COBYLA
result = minimize(cost, x0_scaled, method='COBYLA', constraints=constraints, options=options)

# Rescale the results
opt_scaled = result.x
opt = [opt_scaled[i] * scale_factors[i] for i in range(len(opt_scaled))]

# Output results and check maxcv (maximum constraint violation)
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

    
    
    





    
    example_input_data = np.array(
        [[0.003, 0.007, 500000.0, 800000.0, 0.1]])    
    try:
        results = predict(example_input_data)
        
        # Print the results
        print("Predicted Probabilities:")
        print(results["predicted_probabilities"])
        
        print("\nPredicted Classes:")
        print(results["predicted_classes"])
        
    except Exception as e:
        print(f"An error occurred: {e}")
