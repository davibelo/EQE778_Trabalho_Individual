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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
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
    try:
        logging.info(f"Starting trial {trial.number}")

        # Hyperparameter suggestions
        neurons_ratio = trial.suggest_int('neurons_ratio', 10, 50, step=5)
        dropout_rate_layer1 = trial.suggest_float('dropout_rate_layer1', 0.1, 0.5, step=0.05)
        dropout_rate_layer2 = trial.suggest_float('dropout_rate_layer2', 0.1, 0.5, step=0.05)
        dropout_rate_layer3 = trial.suggest_float('dropout_rate_layer3', 0.1, 0.5, step=0.05)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        regularizer = trial.suggest_float('regularizer', 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])

        logging.info(f"Trial {trial.number} Hyperparameters: "
                     f"neurons_ratio={neurons_ratio}, "
                     f"dropout_rate_layer1={dropout_rate_layer1}, "
                     f"dropout_rate_layer2={dropout_rate_layer2}, "
                     f"dropout_rate_layer3={dropout_rate_layer3}, "
                     f"learning_rate={learning_rate}, "
                     f"regularizer={regularizer}, "
                     f"batch_size={batch_size}")
        
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

        # Compile Model with additional metrics
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        model.compile(
            loss='binary_crossentropy',
            optimizer=opt,
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )

        # Train Model
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=MODEL_CONFIG['patience'], restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(GENERAL_CONFIG['output_folder'], f'best_model_trial{trial.number}.keras'),
                save_best_only=True,
                monitor='val_loss'
            )
        ]

        history = model.fit(
            x_train_scaled,
            y_train_scaled,
            batch_size=batch_size,
            epochs=MODEL_CONFIG['epochs'],
            validation_data=(x_val_scaled, y_val_scaled),
            callbacks=callbacks,
            verbose=0
        )

        # Evaluate on Validation Set - NEW LOGGING SECTION
        evaluation_results = model.evaluate(x_val_scaled, y_val_scaled, verbose=0)
        val_loss, val_accuracy, val_auc, val_precision, val_recall = evaluation_results

        # Log trial metrics
        logging.info(f"Trial {trial.number} Metrics - Validation Loss: {val_loss:.4f}, "
                     f"Accuracy: {val_accuracy:.4f}, AUC: {val_auc:.4f}, "
                     f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")

        return 1 - val_auc  # Optimize for AUC

    except Exception as e:
        logging.error(f"Error in trial {trial.number}: {str(e)}")
        raise

# Run Optuna Study with logging during optimization
logging.info("Starting Optuna study optimization...")
study = optuna.create_study()
study.optimize(objective, n_trials=50)
logging.info("Optuna study optimization completed.")

# Log Best Parameters
best_params = study.best_params
logging.info(f"Best Parameters: {best_params}")

# Save Best Parameters
with open(os.path.join(GENERAL_CONFIG['output_folder'], 'best_params.json'), 'w') as f:
    json.dump(best_params, f)

# Load the Best Model
best_model = tf.keras.models.load_model(os.path.join(GENERAL_CONFIG['output_folder'], f'best_model_trial{study.best_trial.number}.keras'))

# Evaluate Final Model - NEW LOGGING SECTION
evaluation_results = best_model.evaluate(x_test_scaled, y_test_scaled, verbose=0)
test_loss, test_accuracy, test_auc, test_precision, test_recall = evaluation_results

logging.info(f"Final Test Results - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, "
             f"AUC: {test_auc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")

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

# Log metrics
logging.info(f"Final Test Metrics - Accuracy: {accuracy:.4f}, AUC: {roc_auc:.4f}, F1 Score: {f1:.4f}")

# Save metrics to JSON
with open(os.path.join(GENERAL_CONFIG['output_folder'], f"metrics-{MODEL_CONFIG['model_id']}.json"), 'w') as f:
    json.dump(metrics, f)

# Plot Training and Validation Metrics
if 'accuracy' in history.history and 'val_accuracy' in history.history:
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(GENERAL_CONFIG['figures_folder'], f"accuracy-{MODEL_CONFIG['model_id']}.png"))
    plt.close()

if 'loss' in history.history and 'val_loss' in history.history:
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(GENERAL_CONFIG['figures_folder'], f"loss-{MODEL_CONFIG['model_id']}.png"))
    plt.close()
