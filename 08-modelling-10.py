'''
MODELLING - RANDOM FOREST WITH OPTIMIZED PARAMETERS
SEPARATE MODELS FOR EACH OUTPUT
BINARY LABELS
'''
import os
import logging
import joblib
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Dynamically generate the log file name based on the script name
LOG_FILE = f"{os.path.splitext(os.path.basename(__file__))[0]}.log"

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

# Model identification
MODEL_ID = '10'

# Specify data and output folders
INPUT_FOLDER = 'input_files'
OUTPUT_FOLDER = 'output_files'

# Import x and y dataframes
df_scaled_x = joblib.load(os.path.join(INPUT_FOLDER, 'df2_scaled_x.joblib'))
df_scaled_y = joblib.load(os.path.join(INPUT_FOLDER, 'df2_bin_y.joblib'))

x_scaled = df_scaled_x.values
y_scaled = df_scaled_y.values

logging.info(f"x scaled shape: {x_scaled.shape}")
logging.info(f"y scaled shape: {y_scaled.shape}")

# Split data into training and remaining (validation + test) sets
x_train_scaled, x_rem_scaled, y_train_scaled, y_rem_scaled = train_test_split(x_scaled, y_scaled, train_size=0.7, random_state=42)

# Split the remaining data into validation and test sets
x_val_scaled, x_test_scaled, y_val_scaled, y_test_scaled = train_test_split(x_rem_scaled, y_rem_scaled, test_size=1/3, random_state=42)

logging.info(f"x_train shape: {x_train_scaled.shape}")
logging.info(f"y_train shape: {y_train_scaled.shape}")
logging.info(f"x_val shape: {x_val_scaled.shape}")
logging.info(f"y_val shape: {y_val_scaled.shape}")
logging.info(f"x_test shape: {x_test_scaled.shape}")
logging.info(f"y_test shape: {y_test_scaled.shape}")

# Train and optimize separate models for each output
for i in range(y_scaled.shape[1]):
    logging.info(f"Starting optimization for output {i}")
    
    # Extract data for the current output
    y_train = y_train_scaled[:, i]
    y_val = y_val_scaled[:, i]
    y_test = y_test_scaled[:, i]
    
    # Define the objective function specific to the current output
    def objective(trial):
        # Suggest hyperparameters
        n_estimators = trial.suggest_int("n_estimators", 100, 300, step=10)
        max_depth = trial.suggest_int("max_depth", 10, 50, step=10)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        
        # Define the Random Forest model
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        
        # Train the model
        rf_model.fit(x_train_scaled, y_train)
        
        # Evaluate the model on the validation set
        y_val_pred_proba = rf_model.predict_proba(x_val_scaled)[:, 1]
        roc_auc = roc_auc_score(y_val, y_val_pred_proba)
        
        # Log the trial details
        logging.info(
            f"Trial {trial.number}: "
            f"n_estimators={n_estimators}, max_depth={max_depth}, "
            f"min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, "
            f"ROC AUC={roc_auc:.4f}"
        )
        
        return roc_auc

    # Run Optuna optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    
    # Log the best parameters and value
    best_params = study.best_params
    logging.info(f"Best parameters for output {i}: {best_params}")
    logging.info(f"Best validation ROC AUC for output {i}: {study.best_value}")
    
    # Train the final model using the best parameters
    rf_model = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(x_train_scaled, y_train)
    logging.info(f"Final Random Forest model for output {i} training completed.")
    
    # Evaluate the model on the test set
    y_test_pred_proba = rf_model.predict_proba(x_test_scaled)[:, 1]
    y_test_pred_class = rf_model.predict(x_test_scaled)
    
    test_accuracy = accuracy_score(y_test, y_test_pred_class)
    test_roc_auc = roc_auc_score(y_test, y_test_pred_proba)
    
    logging.info(f"Test Accuracy (output {i}): {test_accuracy}")
    logging.info(f"Test ROC AUC (output {i}): {test_roc_auc}")
    logging.info(f"Classification Report (output {i}):\n{classification_report(y_test, y_test_pred_class)}")
    
    # Save the model and its metrics
    joblib.dump(rf_model, os.path.join(OUTPUT_FOLDER, f'opt_model-{MODEL_ID}-output-{i}.joblib'))
    logging.info(f"Optimized Random Forest model for output {i} saved as opt_model-{MODEL_ID}-output-{i}.joblib.")
