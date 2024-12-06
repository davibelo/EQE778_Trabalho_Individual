'''
MODELLING - DEEP LEARNING
WITH HYPERPARAMETER OPTIMIZATION
'''

import io
import os
import logging
import joblib
import optuna
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, regularizers
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from optuna.integration import TFKerasPruningCallback


# Dynamically generate the log file name based on the script name
LOG_FILE = f"{os.path.splitext(os.path.basename(__file__))[0]}.log"

# Configure logging
class FlushableFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(
    handlers=[FlushableFileHandler(LOG_FILE, mode='w')],
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# Model identification
MODEL_ID = '06'

# Specify the folder to export the figures
FIGURES_FOLDER = 'figures'

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

def objective(trial):
    # Clear the previous TensorFlow session
    tf.keras.backend.clear_session()

    # Define the search space for hyperparameters
    num_dense_layers = trial.suggest_int('num_dense_layers', 2, 5)
    neurons_ratio = trial.suggest_float('neurons_ratio', 2, 5)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    activation = trial.suggest_categorical('activation', ['relu', 'softplus', 'tanh'])

    # Log the hyperparameters for the current trial
    logging.info(f"Trial {trial.number} hyperparameters:")
    logging.info(f"num_dense_layers={num_dense_layers}, neurons_ratio={neurons_ratio}")
    logging.info(f"dropout_rate={dropout_rate}, learning_rate={learning_rate}, activation={activation}")

    # Define the model
    num_features = x_train_scaled.shape[1]
    num_outputs = y_train_scaled.shape[1]

    # Initial number of neurons for the first layer
    neurons = int(neurons_ratio * num_features)
    neurons = 2 * ((neurons + 1) // 2)  # Round to nearest multiple of 2
   
    # Build the model
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(num_features,)))
    logging.info(f"Adding Input layer with {num_features} neurons")

    # Add hidden layers (exactly num_dense_layers)
    for i in range(num_dense_layers):
        logging.info(f"Adding Dense layer {i + 1} with {neurons} neurons")
        model.add(
            layers.Dense(
                neurons,
                activation=activation,
                kernel_regularizer=regularizers.l2(0.01)
            )
        )
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.BatchNormalization())
        neurons = max(2, 2 * ((neurons // 2 + 1) // 2))  # Ensure neurons are multiple of 2

    # Log after hidden layers
    logging.info("Finished adding hidden layers.")

    # Add the output layer
    logging.info(f"Adding output Dense layer with {num_outputs} neurons")
    model.add(layers.Dense(num_outputs))

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse')

    # Log model summary with detailed layer information
    buffer = io.StringIO()
    model.summary(print_fn=lambda x: buffer.write(x + "\n"))
    summary = buffer.getvalue()
    buffer.close()
    logging.info("Model Summary:")
    for line in summary.splitlines():
        logging.info(line)
    logging.info("Detailed Layer Information:")
    for i, layer in enumerate(model.layers):
        try:
            output_shape = layer.output_shape  # Accessible after model is built
        except AttributeError:
            output_shape = "Not built yet"

        layer_info = (
            f"Layer {i + 1}: {layer.name}, "
            f"Type: {layer.__class__.__name__}, "
            f"Output Shape: {output_shape}, "
            f"Parameters: {layer.count_params()}"
        )
        logging.info(layer_info)
   
    # Train the model
    history = model.fit(
        x_train_scaled, y_train_scaled,
        validation_data=(x_val_scaled, y_val_scaled),
        epochs=100,
        batch_size=5000,
        callbacks=[TFKerasPruningCallback(trial, 'val_loss')],
        verbose=1
    )

    # Evaluate the model
    val_loss = model.evaluate(x_val_scaled, y_val_scaled, verbose=0)
    logging.info(f"Trial {trial.number}: Validation Loss: {val_loss}")
    trial.set_user_attr("final_model", model)
    return val_loss


def trial_callback(study, trial):
    logging.info(
        f"Trial {trial.number} completed with value: {trial.value}. "
        f"Parameters: {trial.params}"
    )


# Create Optuna study and optimize
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, callbacks=[trial_callback])

# Log the best hyperparameters
best_trial = study.best_trial
logging.info(f"Best trial value: {best_trial.value}")
logging.info(f"Best trial params: {best_trial.params}")

# Retrieve the best model
best_model = best_trial.user_attrs["final_model"]

# Save the best model
best_model_path = os.path.join(OUTPUT_FOLDER, f'{MODEL_ID}_best.keras')
best_model.save(best_model_path)
logging.info(f"Best model saved to {best_model_path}")

# Visualize optimization results
try:
    import optuna.visualization as vis
    vis.plot_optimization_history(study).write_html(
        os.path.join(FIGURES_FOLDER, f'{MODEL_ID}_optimization_history.html')
    )
    vis.plot_param_importances(study).write_html(
        os.path.join(FIGURES_FOLDER, f'{MODEL_ID}_param_importance.html')
    )
    logging.info("Saved Optuna visualizations.")
except ImportError:
    logging.warning("Optuna visualization libraries are not installed.")

# Retrieve the best model
best_model = best_trial.user_attrs["final_model"]

# Evaluate the best model on test data
test_loss = best_model.evaluate(x_test_scaled, y_test_scaled, verbose=0)
logging.info(f"Test loss of the best model: {test_loss}")

# Save the best model
best_model.save(os.path.join(OUTPUT_FOLDER, f'{MODEL_ID}_best.keras'))

#==============================================================================

# Plot the optimization history
optuna.visualization.plot_optimization_history(study).write_html(
    os.path.join(FIGURES_FOLDER, f'{MODEL_ID}_optimization_history.html')
)
optuna.visualization.plot_param_importances(study).write_html(
    os.path.join(FIGURES_FOLDER, f'{MODEL_ID}_param_importance.html')
)

# Assuming you have your model predictions and actual values
y_test_pred_scaled = model.predict(x_test_scaled)
y_pred_scaled = model.predict(x_scaled)
residue_test = y_test_scaled.ravel() - y_test_pred_scaled.ravel()
residue = y_scaled.ravel() - y_pred_scaled.ravel()

# Calculate RMSE
rmse_dl = np.sqrt(model.evaluate(x_test_scaled, y_test_scaled, batch_size=1000, verbose=0))
logging.info(f"RMSE: {rmse_dl}")

# Calculate R-squared value
r2_test = r2_score(y_test_scaled, y_test_pred_scaled)
logging.info(f"Test data R²: {r2_test}")

r2_all = r2_score(y_scaled, y_pred_scaled)
logging.info(f"All data R²: {r2_all}")

plt.figure(figsize = (10,5))
plt.plot(history.history['loss'], linewidth = 2)
plt.plot(history.history['val_loss'], linewidth = 2)
plt.title('Training loss RMSE: %.3f' %(rmse_dl), fontsize = 14)
plt.ylabel('Loss', fontsize = 12)
plt.xlabel('Epoch', fontsize = 12)
plt.legend(['train', 'validation'], loc='upper right', fontsize = 12)
plt.grid()
plt.xticks(fontsize = 11)
plt.yticks(fontsize = 11)
plt.savefig(f'{FIGURES_FOLDER}/model_{MODEL_ID}_training_loss.png')

prediction_results = np.asarray(np.column_stack((y_test_scaled, y_test_pred_scaled)))
prediction_results = sorted(prediction_results, key= lambda x: x[0])
prediction_results = np.asarray(prediction_results)
plt.figure(figsize = (10,5))
plt.plot(np.asarray(np.asarray(prediction_results[:,0])), 
         linestyle='-', linewidth=0.5, marker='o', markersize=3, label = 'Real')
plt.plot(np.asarray(np.asarray(prediction_results[:,1])), 
          linestyle='-', linewidth=0.5, marker='^', markersize=3, label = 'Prediction')
plt.title('Test Data: y real x y prediction, RMSE: %.3f'
          %(rmse_dl), fontsize = 14)
plt.xlabel('Test Data', fontsize = 12)
plt.ylabel('y normalized', fontsize = 12)
plt.legend(fontsize = 12)
plt.grid()
plt.xticks(fontsize = 11)
plt.yticks(fontsize = 11)
plt.savefig(f'{FIGURES_FOLDER}/model_{MODEL_ID}_test_data_real_and_pred.png')

prediction_results = np.asarray(np.column_stack((y_scaled, y_pred_scaled)))
prediction_results = sorted(prediction_results, key= lambda x: x[0])
prediction_results = np.asarray(prediction_results)
plt.figure(figsize = (10,5))
plt.plot(np.asarray(np.asarray(prediction_results[:,0])), 
         linestyle='-', linewidth=0.5, marker='o', markersize=3, label = 'Real')
plt.plot(np.asarray(np.asarray(prediction_results[:,1])), 
         linestyle='-', linewidth=0.5, marker='^', markersize=3, label = 'Prediction')
plt.title('All Data: y real x y prediction, RMSE: %.3f'
          %(rmse_dl), fontsize = 14)
plt.xlabel('All Data', fontsize = 12)
plt.ylabel('y normalized', fontsize = 12)
plt.legend(fontsize = 12)
plt.grid()
plt.xticks(fontsize = 11)
plt.yticks(fontsize = 11)
plt.savefig(f'{FIGURES_FOLDER}/model_{MODEL_ID}_all_data_real_and_pred.png')

# Predicted vs True values - Test Data
plt.figure()
reta = np.random.uniform(low=-2, high=2, size=(50,))
plt.plot(reta,reta, color='black', label='x = y') #plot reta x = y
plt.scatter(y_test_scaled, y_test_pred_scaled, color='blue', marker='x')
plt.legend(fontsize=15, loc='best')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predicted Values', fontsize=15)
plt.title('Prediction - Test Data', fontsize=15)
plt.tight_layout()
plt.savefig(f'{FIGURES_FOLDER}/model_{MODEL_ID}_test_real_vs_pred.png')

# Residue - Test Data
plt.figure()
plt.grid(axis='y')
plt.hist(x=residue_test, bins='auto', ec='black')
plt.ylabel('Frequency', fontsize=15)
plt.xlabel('Residue', fontsize=15)
plt.title('Residue - Test Data', fontsize=15)
plt.tight_layout()
plt.savefig(f'{FIGURES_FOLDER}/MODEL_{MODEL_ID}_test_residue.png')

plt.close()

