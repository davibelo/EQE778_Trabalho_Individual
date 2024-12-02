import os
import logging
import joblib
import numpy as np
import optuna
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

# Optuna objective function
def objective(trial):
    # Suggest hyperparameters
    n_estimators = trial.suggest_int("n_estimators", 100, 300)
    max_depth = trial.suggest_int("max_depth", 10, 50)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
    
    # Define the Random Forest model
    rf_base_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )
    
    multi_rf_model = MultiOutputRegressor(rf_base_model)
    
    # Train the model
    multi_rf_model.fit(x_train_scaled, y_train_scaled)
    
    # Evaluate the model on the validation set
    y_val_pred = multi_rf_model.predict(x_val_scaled)
    avg_mse = np.mean(mean_squared_error(y_val_scaled, y_val_pred, multioutput='raw_values'))    
    avg_r2 = np.mean(r2_score(y_val_scaled, y_val_pred, multioutput='raw_values'))

    # Log the trial details
    logging.info(
        f"Trial {trial.number}: "
        f"n_estimators={n_estimators}, max_depth={max_depth}, "
        f"min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, "
        f"Avg MSE={avg_mse:.4f}, "
        f"Avg R2={avg_r2:.4f}"
    )
    
    return avg_r2

# Run Optuna optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)  # Adjust `n_trials` for more extensive optimization

# Log the best parameters and value
logging.info(f"Best parameters: {study.best_params}")
logging.info(f"Best validation: {study.best_value}")

# Train the final model using the best parameters
best_params = study.best_params

rf_base_model = RandomForestRegressor(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    min_samples_split=best_params["min_samples_split"],
    min_samples_leaf=best_params["min_samples_leaf"],
    random_state=42,
    n_jobs=-1
)

multi_rf_model = MultiOutputRegressor(rf_base_model)
multi_rf_model.fit(x_train_scaled, y_train_scaled)
logging.info("Final Multi-output Random Forest Regressor training completed.")

# Evaluate the model on the test set
y_test_pred = multi_rf_model.predict(x_test_scaled)

# Calculate metrics for each output
test_mse = mean_squared_error(y_test_scaled, y_test_pred, multioutput='raw_values')
test_mae = mean_absolute_error(y_test_scaled, y_test_pred, multioutput='raw_values')
test_r2  = r2_score(y_test_scaled, y_test_pred, multioutput='raw_values')

for i, (mse, mae, r2) in enumerate(zip(test_mse, test_mae, test_r2)):
    logging.info(f"Test MSE (output {i}): {mse}")
    logging.info(f"Test MAE (output {i}): {mae}")
    logging.info(f"Test R2 (output {i}): {r2}")

# Save the optimized model
joblib.dump(multi_rf_model, os.path.join(OUTPUT_FOLDER, f'optimized_multi_model-{MODEL_ID}.joblib'))
logging.info(f"Optimized Multi-output Random Forest Regressor model saved as optimized_multi_model-{MODEL_ID}.joblib.")

# Plot the True vs. Predicted values for the test set
plt.figure()
reta = np.random.uniform(low=-8.5, high=5, size=(50,))
plt.plot(reta,reta, color='black', label='x = y') #plot reta x = y
plt.scatter(y_test_scaled, y_test_pred, color='blue', marker='x')
plt.legend(fontsize=15, loc='best')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predicted Values', fontsize=15)
plt.title('Prediction - RF - Test', fontsize=15)
plt.tight_layout()
plt.savefig(f'{FIGURES_FOLDER}/model_{MODEL_ID}_testpredictions_eng.png')
plt.close()

# Gráfico Resíduos ingles - teste
residue = y_test_scaled.ravel() - y_test_pred.ravel()
plt.figure()
plt.grid(axis='y')
plt.hist(x=residue, bins='auto', ec='black')
plt.ylabel('Frequency', fontsize=15)
plt.xlabel('Residue', fontsize=15)
plt.title('Residue - RF - Test', fontsize=15)
plt.tight_layout()
plt.savefig(f'{FIGURES_FOLDER}/model_{MODEL_ID}_testresidue_eng.png')
plt.close()