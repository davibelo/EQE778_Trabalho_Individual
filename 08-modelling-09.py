'''
MODELLING - RANDOM FOREST WITH OPTIMIZED PARAMETERS
BINARY LABELS
'''
import os
import logging
import joblib
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
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
MODEL_ID = '09'

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

# Optuna objective function
def objective(trial):
    # Suggest hyperparameters
    n_estimators = trial.suggest_int("n_estimators", 100, 300, step=10)
    max_depth = trial.suggest_int("max_depth", 10, 50, step=10)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    
    # Define the Random Forest model
    rf_base_model = RandomForestClassifier(        
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )
    
    multi_rf_model = MultiOutputClassifier(rf_base_model)
    
    # Train the model
    multi_rf_model.fit(x_train_scaled, y_train_scaled)
    
    # Evaluate the model on the validation set
    y_val_pred_proba = np.column_stack([estimator.predict_proba(x_val_scaled)[:, 1] for estimator in multi_rf_model.estimators_])
    
    # Calculate ROC AUC for each output and average them
    roc_aucs = [roc_auc_score(y_val_scaled[:, i], y_val_pred_proba[:, i]) for i in range(y_val_scaled.shape[1])]
    avg_roc_auc = np.mean(roc_aucs)
    
    # Log the trial details
    logging.info(
        f"Trial {trial.number}: "
        f"n_estimators={n_estimators}, max_depth={max_depth}, "
        f"min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, "
        f"Avg ROC AUC={avg_roc_auc:.4f}"
    )
    
    return avg_roc_auc

# Run Optuna optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)  # Adjust `n_trials` for more extensive optimization

# Log the best parameters and value
logging.info(f"Best parameters: {study.best_params}")
logging.info(f"Best validation ROC AUC: {study.best_value}")

# Train the final model using the best parameters
best_params = study.best_params

rf_base_model = RandomForestClassifier(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    min_samples_split=best_params["min_samples_split"],
    min_samples_leaf=best_params["min_samples_leaf"],
    random_state=42,
    n_jobs=-1
)

multi_rf_model = MultiOutputClassifier(rf_base_model)
multi_rf_model.fit(x_train_scaled, y_train_scaled)
logging.info("Final Multi-output Random Forest model training completed.")

# Evaluate the model on the test set
y_test_pred_proba = np.column_stack([estimator.predict_proba(x_test_scaled)[:, 1] for estimator in multi_rf_model.estimators_])
y_test_pred_class = multi_rf_model.predict(x_test_scaled)

test_accuracies = [accuracy_score(y_test_scaled[:, i], y_test_pred_class[:, i]) for i in range(y_test_scaled.shape[1])]
test_roc_aucs = [roc_auc_score(y_test_scaled[:, i], y_test_pred_proba[:, i]) for i in range(y_test_scaled.shape[1])]
test_f1_scores = [f1_score(y_test_scaled[:, i], y_test_pred_class[:, i], average='weighted') for i in range(y_test_scaled.shape[1])]

for i, (acc, roc, f1) in enumerate(zip(test_accuracies, test_roc_aucs, test_f1_scores)):
    logging.info(f"Test Accuracy (output {i}): {acc}")
    logging.info(f"Test ROC AUC (output {i}): {roc}")
    logging.info(f"Test F1 Score (output {i}, weighted): {f1}")
    logging.info(f"Classification Report (output {i}):\n{classification_report(y_test_scaled[:, i], y_test_pred_class[:, i])}")
for i, (acc, roc) in enumerate(zip(test_accuracies, test_roc_aucs)):
    logging.info(f"Test Accuracy (output {i}): {acc}")
    logging.info(f"Test ROC AUC (output {i}): {roc}")
    logging.info(f"Classification Report (output {i}):\n{classification_report(y_test_scaled[:, i], y_test_pred_class[:, i])}")

# Save the optimized model
joblib.dump(multi_rf_model, os.path.join(OUTPUT_FOLDER, f'optimized_multi_model-{MODEL_ID}.joblib'))
logging.info(f"Optimized Multi-output Random Forest model saved as optimized_multi_model-{MODEL_ID}.joblib.")
