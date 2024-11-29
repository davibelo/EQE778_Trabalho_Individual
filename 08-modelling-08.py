'''
MODELLING - RANDOM FOREST
BINARY LABELS
'''
import os
import logging
import joblib
import numpy as np
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
    format='%(asctime)s - %(message)s',
    filemode='w'
)

# Model identification
MODEL_ID = '08'

# Specify the folder to export the figures
FIGURES_FOLDER = 'figures'

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
x_train_scaled, x_rem_scaled, y_train_scaled, y_rem_scaled = train_test_split(x_scaled, y_scaled, train_size=0.9, random_state=42)

# Split the remaining data into validation and test sets
x_val_scaled, x_test_scaled, y_val_scaled, y_test_scaled = train_test_split(x_rem_scaled, y_rem_scaled, test_size=1/3, random_state=42)

logging.info(f"x_train shape: {x_train_scaled.shape}")
logging.info(f"y_train shape: {y_train_scaled.shape}")
logging.info(f"x_val shape: {x_val_scaled.shape}")
logging.info(f"y_val shape: {y_val_scaled.shape}")
logging.info(f"x_test shape: {x_test_scaled.shape}")
logging.info(f"y_test shape: {y_test_scaled.shape}")

# Define the Random Forest model using MultiOutputClassifier
rf_base_model = RandomForestClassifier(
    n_estimators=100,          # Number of trees in the forest
    max_depth=None,            # Maximum depth of the tree
    min_samples_split=2,       # Minimum samples required to split a node
    min_samples_leaf=1,        # Minimum samples required at each leaf node
    random_state=42,           # Random state for reproducibility
    n_jobs=-1                  # Use all available cores for training
)

multi_rf_model = MultiOutputClassifier(rf_base_model)

# Train the Random Forest model
multi_rf_model.fit(x_train_scaled, y_train_scaled)
logging.info("Multi-output Random Forest model training completed.")

# Evaluate the model on the validation set
y_val_pred_proba = np.column_stack([estimator.predict_proba(x_val_scaled)[:, 1] for estimator in multi_rf_model.estimators_])
y_val_pred_class = multi_rf_model.predict(x_val_scaled)

val_accuracies = [accuracy_score(y_val_scaled[:, i], y_val_pred_class[:, i]) for i in range(y_val_scaled.shape[1])]
val_roc_aucs = [roc_auc_score(y_val_scaled[:, i], y_val_pred_proba[:, i]) for i in range(y_val_scaled.shape[1])]

for i, (acc, roc) in enumerate(zip(val_accuracies, val_roc_aucs)):
    logging.info(f"Validation Accuracy (output {i}): {acc}")
    logging.info(f"Validation ROC AUC (output {i}): {roc}")

# Evaluate the model on the test set
y_test_pred_proba = np.column_stack([estimator.predict_proba(x_test_scaled)[:, 1] for estimator in multi_rf_model.estimators_])
y_test_pred_class = multi_rf_model.predict(x_test_scaled)

test_accuracies = [accuracy_score(y_test_scaled[:, i], y_test_pred_class[:, i]) for i in range(y_test_scaled.shape[1])]
test_roc_aucs = [roc_auc_score(y_test_scaled[:, i], y_test_pred_proba[:, i]) for i in range(y_test_scaled.shape[1])]

for i, (acc, roc) in enumerate(zip(test_accuracies, test_roc_aucs)):
    logging.info(f"Test Accuracy (output {i}): {acc}")
    logging.info(f"Test ROC AUC (output {i}): {roc}")
    logging.info(f"Classification Report (output {i}):\n{classification_report(y_test_scaled[:, i], y_test_pred_class[:, i])}")

# Save the Random Forest model
joblib.dump(multi_rf_model, os.path.join(OUTPUT_FOLDER, f'multi_model-{MODEL_ID}.joblib'))
logging.info(f"Multi-output Random Forest model saved as multi_model-{MODEL_ID}.joblib.")

# Load the saved model for testing
loaded_model = joblib.load(os.path.join(OUTPUT_FOLDER, f'multi_model-{MODEL_ID}.joblib'))

# Predict using the loaded model
y_test_pred_class_loaded = loaded_model.predict(x_test_scaled)
loaded_accuracies = [accuracy_score(y_test_scaled[:, i], y_test_pred_class_loaded[:, i]) for i in range(y_test_scaled.shape[1])]

for i, acc in enumerate(loaded_accuracies):
    logging.info(f"Loaded model test accuracy (output {i}): {acc}")
