'''
MODELLING - DEEP LEARNING version c
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
from tensorflow.keras import layers, regularizers
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import optuna
import optuna

# General Configuration
GENERAL_CONFIG = {
    'figures_folder': 'figures',
    'input_folder': 'input_files',
    'output_folder': 'output_files'
}

# Model Configuration
MODEL_CONFIG = {
    'model_id': '07c',
    'patience': 20,
    'epochs': 100,
    'multiple': 2
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
df_scaled_x = joblib.load(os.path.join(GENERAL_CONFIG['input_folder'], 'df2_scaled_x.joblib'))
df_scaled_y = joblib.load(os.path.join(GENERAL_CONFIG['input_folder'], 'df2_bin_y.joblib'))

x_scaled = df_scaled_x.values
y_scaled = df_scaled_y.values

# Split data
x_train_scaled, x_rem_scaled, y_train_scaled, y_rem_scaled = train_test_split(x_scaled, y_scaled, train_size=0.7, random_state=42)
x_val_scaled, x_test_scaled, y_val_scaled, y_test_scaled = train_test_split(x_rem_scaled, y_rem_scaled, test_size=1/3, random_state=42)

# Function to determine neurons
def neurons(num_features, ratio, multiple):
    neuron_count = int(num_features * ratio)
    return max(multiple, round(neuron_count / multiple) * multiple)

# Objective function for Optuna
def objective(trial):
    # Hyperparameter suggestions
    neurons_ratio = trial.suggest_int('neurons_ratio', 10, 50, step=5)
    dropout_rate_layer1 = trial.suggest_float('dropout_rate_layer1', 0.1, 0.5, step=0.05)
    dropout_rate_layer2 = trial.suggest_float('dropout_rate_layer2', 0.1, 0.5, step=0.05)
    dropout_rate_layer3 = trial.suggest_float('dropout_rate_layer3', 0.1, 0.5, step=0.05)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    regularizer = trial.suggest_float('regularizer', 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])

    # Define Model
    num_features = x_train_scaled.shape[1]
    num_outputs = y_train_scaled.shape[1]

    model = tf.keras.Sequential([
        layers.Input(shape=(num_features,)),
        layers.Dense(neurons(num_features, neurons_ratio, MODEL_CONFIG['multiple']), kernel_regularizer=regularizers.l2(regularizer)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(dropout_rate_layer1),

        layers.Dense(neurons(num_features, neurons_ratio / 2, MODEL_CONFIG['multiple']), kernel_regularizer=regularizers.l2(regularizer)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(dropout_rate_layer2),

        layers.Dense(neurons(num_features, neurons_ratio / 4, MODEL_CONFIG['multiple']), kernel_regularizer=regularizers.l2(regularizer)),
        layers.ReLU(),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate_layer3),

        layers.Dense(num_outputs, activation='sigmoid')
    ])

    # Compile Model
    opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    model.compile(
        loss='binary_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=MODEL_CONFIG['patience'], restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(GENERAL_CONFIG['output_folder'], 'best_model.h5'),
            save_best_only=True,
            monitor='val_loss'
        )
    ]

    # Train Model
    history = model.fit(
        x_train_scaled,
        y_train_scaled,
        batch_size=batch_size,
        epochs=MODEL_CONFIG['epochs'],
        validation_data=(x_val_scaled, y_val_scaled),
        callbacks=callbacks,
        verbose=0
    )

    # Evaluate Model
    val_accuracy = max(history.history['val_accuracy'])
    return 1 - val_accuracy  # Optuna minimizes the objective

# Run Optuna Study
study = optuna.create_study()
study.optimize(objective, n_trials=50)

# Log Best Parameters
best_params = study.best_params
logging.info(f"Best Parameters: {best_params}")

# Save Best Parameters
with open(os.path.join(GENERAL_CONFIG['output_folder'], 'best_params.json'), 'w') as f:
    json.dump(best_params, f)

# Load the Best Model
best_model = tf.keras.models.load_model(os.path.join(GENERAL_CONFIG['output_folder'], 'best_model.h5'))

# Evaluate Final Model
evaluation_results = best_model.evaluate(x_test_scaled, y_test_scaled)
logging.info(f"Final Test Results: {evaluation_results}")

# Predictions
y_test_pred_proba = best_model.predict(x_test_scaled)
y_test_pred_class = (y_test_pred_proba > 0.5).astype(int)

# Metrics
accuracy = accuracy_score(y_test_scaled, y_test_pred_class)
roc_auc = roc_auc_score(y_test_scaled, y_test_pred_proba)
f1 = f1_score(y_test_scaled, y_test_pred_class, average='macro')

metrics = {
    'test_loss': evaluation_results[0],
    'test_accuracy': evaluation_results[1],
    'roc_auc': roc_auc,
    'f1_score': f1
}

with open(os.path.join(GENERAL_CONFIG['output_folder'], f"metrics-{MODEL_CONFIG['model_id']}.json"), 'w') as f:
    json.dump(metrics, f)

# Plot Training and Validation Metrics
history = best_model.history
plt.figure()
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(GENERAL_CONFIG['figures_folder'], f"accuracy-{MODEL_CONFIG['model_id']}.png"))

plt.figure()
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(GENERAL_CONFIG['figures_folder'], f"loss-{MODEL_CONFIG['model_id']}.png"))
