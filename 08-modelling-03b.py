import os
import optuna
import logging
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
MODEL_ID = '03b'

# Specify data and output folders
INPUT_FOLDER = 'input_files'
OUTPUT_FOLDER = 'output_files'
FIGURES_FOLDER = 'figures'

# Import x and y dataframes
df_scaled_x = joblib.load(os.path.join(INPUT_FOLDER, 'df_scaled_x.joblib'))
df_scaled_y = joblib.load(os.path.join(INPUT_FOLDER, 'df_scaled_y.joblib'))

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

# Train and evaluate separate models with Optuna optimization for each output
separate_models = []

for i in range(y_train_scaled.shape[1]):
    logging.info(f"Starting Optuna optimization for output {i + 1}")
    
    # Define the Optuna objective function for the current output
    def objective(trial):
        # Suggest hyperparameters
        n_estimators = trial.suggest_int("n_estimators", 100, 300)
        max_depth = trial.suggest_int("max_depth", 10, 50)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
        
        # Define the Random Forest model
        rf_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        
        # Train the model
        rf_model.fit(x_train_scaled, y_train_scaled[:, i])
        
        # Evaluate the model on the validation set
        y_val_pred = rf_model.predict(x_val_scaled)
        mse = mean_squared_error(y_val_scaled[:, i], y_val_pred)
        r2 = r2_score(y_val_scaled[:, i], y_val_pred)

        logging.info(
            f"Output {i + 1} - Trial {trial.number}: "
            f"n_estimators={n_estimators}, max_depth={max_depth}, "
            f"min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, "
            f"MSE={mse:.4f}, R2={r2:.4f}"
        )
        
        return r2  # Optimize R2 for the current output

    # Create and optimize the study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)  # Adjust `n_trials` for more thorough optimization

    # Log the best parameters and their value
    logging.info(f"Output {i + 1} - Best parameters: {study.best_params}")
    logging.info(f"Output {i + 1} - Best validation: {study.best_value}")

    # Train the final model using the best parameters for the current output
    best_params = study.best_params
    rf_model = RandomForestRegressor(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(x_train_scaled, y_train_scaled[:, i])
    separate_models.append(rf_model)

    # Evaluate the model on the test set
    y_test_pred = rf_model.predict(x_test_scaled)
    mse = mean_squared_error(y_test_scaled[:, i], y_test_pred)
    mae = mean_absolute_error(y_test_scaled[:, i], y_test_pred)
    r2 = r2_score(y_test_scaled[:, i], y_test_pred)

    logging.info(f"Output {i + 1} - Test MSE: {mse}")
    logging.info(f"Output {i + 1} - Test MAE: {mae}")
    logging.info(f"Output {i + 1} - Test R2: {r2}")

    # Save the optimized model for this output
    model_path = os.path.join(OUTPUT_FOLDER, f'optimized_model_output_{i + 1}-{MODEL_ID}.joblib')
    joblib.dump(rf_model, model_path)
    logging.info(f"Model for output {i + 1} saved as {model_path}.")

    # Plot True vs Predicted values
    plt.figure()
    plt.scatter(y_test_scaled[:, i], y_test_pred, color='blue', marker='x', label=f'Output {i + 1}')
    plt.plot([y_test_scaled[:, i].min(), y_test_scaled[:, i].max()],
             [y_test_scaled[:, i].min(), y_test_scaled[:, i].max()],
             color='black', label='x = y')
    plt.legend(fontsize=15, loc='best')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predicted Values', fontsize=15)
    plt.title(f'Prediction - RF - Test (Output {i + 1})', fontsize=15)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_FOLDER}/model_{MODEL_ID}_testpredictions_output_{i + 1}_eng.png')
    plt.close()

    # Plot Residuals
    residue = y_test_scaled[:, i] - y_test_pred
    plt.figure()
    plt.grid(axis='y')
    plt.hist(x=residue, bins='auto', ec='black')
    plt.ylabel('Frequency', fontsize=15)
    plt.xlabel('Residue', fontsize=15)
    plt.title(f'Residue - RF - Test (Output {i + 1})', fontsize=15)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_FOLDER}/model_{MODEL_ID}_testresidue_output_{i + 1}_eng.png')
    plt.close()