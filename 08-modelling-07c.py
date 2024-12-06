'''
MODELLING - DEEP LEARNING version b
BINARY LABELS
'''
import os
import time
import json
import logging
import joblib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import optuna
from tensorflow.keras import layers, regularizers
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# Configuration Block
CONFIG = {
    'folders': {
        'figures_folder': 'figures',
        'input_folder': 'input_files',
        'output_folder': 'output_files',
    },
    'model_id': '07b',
    'training': {
        'patience': 20,
        'epochs': 100,
    }
}

# Dynamically generate the log file name based on the script name
LOG_FILE = f"{os.path.splitext(os.path.basename(__file__))[0]}.log"

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

# Load Data
df_scaled_x = joblib.load(os.path.join(CONFIG['folders']['input_folder'], 'df2_scaled_x.joblib'))
df_scaled_y = joblib.load(os.path.join(CONFIG['folders']['input_folder'], 'df2_bin_y.joblib'))

x_scaled = df_scaled_x.values
y_scaled = df_scaled_y.values

logging.info(f"x scaled shape: {x_scaled.shape}")
logging.info(f"y scaled shape: {y_scaled.shape}")

# Split data
x_train_scaled, x_rem_scaled, y_train_scaled, y_rem_scaled = train_test_split(x_scaled, y_scaled, train_size=0.7, random_state=42)
x_val_scaled, x_test_scaled, y_val_scaled, y_test_scaled = train_test_split(x_rem_scaled, y_rem_scaled, test_size=1/3, random_state=42)

logging.info(f"x_train shape: {x_train_scaled.shape}")
logging.info(f"y_train shape: {y_train_scaled.shape}")
logging.info(f"x_val shape: {x_val_scaled.shape}")
logging.info(f"y_val shape: {y_val_scaled.shape}")
logging.info(f"x_test shape: {x_test_scaled.shape}")
logging.info(f"y_test shape: {y_test_scaled.shape}")

# Function to define a model for hyperparameter tuning
def create_model(trial):
    num_features = x_train_scaled.shape[1]
    num_outputs = y_train_scaled.shape[1]

    # Hyperparameters to tune
    neurons_ratio = trial.suggest_float('neurons_ratio', 10, 50)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'sigmoid'])
    l1_value = trial.suggest_float('l1_value', 1e-5, 1e-2)
    l2_value = trial.suggest_float('l2_value', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])

    reg = regularizers.L1L2(l1=l1_value, l2=l2_value)

    model = tf.keras.Sequential([
        layers.Input(shape=(num_features,)),
        layers.Dense(int(num_features * neurons_ratio), kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation(activation),
        layers.Dropout(dropout_rate),

        layers.Dense(int(num_features * neurons_ratio / 2), kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation(activation),
        layers.Dropout(dropout_rate / 2),

        layers.Dense(int(num_features * neurons_ratio / 4), kernel_regularizer=reg),
        layers.Activation(activation),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate / 4),

        layers.Dense(num_outputs, activation='sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    return model, batch_size

# Objective function for Optuna
def objective(trial):
    model, batch_size = create_model(trial)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=CONFIG['training']['patience'], restore_best_weights=True)
    ]

    history = model.fit(
        x_train_scaled, y_train_scaled,
        batch_size=batch_size,
        epochs=50,
        validation_data=(x_val_scaled, y_val_scaled),
        callbacks=callbacks,
        verbose=0
    )

    val_loss = min(history.history['val_loss'])
    logging.info(f"Trial {trial.number}: Validation Loss: {val_loss}")
    return val_loss

# Run Optuna study
study = optuna.create_study(direction='minimize')
logging.info("Starting Optuna study...")
study.optimize(objective, n_trials=50, callbacks=[
    lambda study, trial: logging.info(f"Trial {trial.number} completed with value: {trial.value}, params: {trial.params}")
])
logging.info("Optuna study completed.")

# Log the best parameters
logging.info(f"Best trial: {study.best_trial.params}")

# Train final model with best parameters
best_params = study.best_trial.params
final_model, batch_size = create_model(optuna.trial.FixedTrial(best_params))

history = final_model.fit(
    x_train_scaled, y_train_scaled,
    batch_size=batch_size,
    epochs=CONFIG['training']['epochs'],
    validation_data=(x_val_scaled, y_val_scaled),
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=CONFIG['training']['patience'], restore_best_weights=True)]
)

# Save the final model
os.makedirs(CONFIG['folders']['output_folder'], exist_ok=True)
final_model.save(os.path.join(CONFIG['folders']['output_folder'], f"best_model-{CONFIG['model_id']}.keras"))

# Plot Training and Validation Metrics
os.makedirs(CONFIG['folders']['figures_folder'], exist_ok=True)
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(CONFIG['folders']['figures_folder'], f"accuracy-{CONFIG['model_id']}.png"))

plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(CONFIG['folders']['figures_folder'], f"loss-{CONFIG['model_id']}.png"))
