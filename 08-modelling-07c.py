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
from tensorflow.keras import layers, regularizers
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# Configuration Block
CONFIG = {
    'neurons_ratio': 20,
    'dropout_rate': 0.05,
    'batch_size': 128,
    'learning_rate': 0.001,
    'patience': 20,
    'epochs': 100,
    'multiple': 2,
    'figures_folder': 'figures',
    'input_folder': 'input_files',
    'output_folder': 'output_files',
    'model_id': '07c'
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
df_scaled_x = joblib.load(os.path.join(CONFIG['input_folder'], 'df2_scaled_x.joblib'))
df_scaled_y = joblib.load(os.path.join(CONFIG['input_folder'], 'df2_bin_y.joblib'))

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

# # Handle Class Imbalance
# class_weights = {
#     0: (1 / np.sum(y_train_scaled == 0)) * (len(y_train_scaled) / 2.0),
#     1: (1 / np.sum(y_train_scaled == 1)) * (len(y_train_scaled) / 2.0)
# }

# Function to determine neurons
def neurons(num_features, ratio, multiple=CONFIG['multiple']):
    neuron_count = int(num_features * ratio)
    return max(multiple, round(neuron_count / multiple) * multiple)

# Define Model
num_features = x_train_scaled.shape[1]
num_outputs = y_train_scaled.shape[1]

model = tf.keras.Sequential([
    layers.Input(shape=(num_features,)),
    layers.Dense(neurons(num_features, CONFIG['neurons_ratio']), kernel_regularizer=regularizers.l2(0.0001)),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Dropout(0.3),

    layers.Dense(neurons(num_features, CONFIG['neurons_ratio'] / 2), kernel_regularizer=regularizers.l2(0.0001)),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Dropout(0.2),

    layers.Dense(neurons(num_features, CONFIG['neurons_ratio'] / 4), kernel_regularizer=regularizers.l2(0.0001)),
    layers.ReLU(),
    layers.BatchNormalization(),
    layers.Dropout(0.1),

    layers.Dense(num_outputs, activation='sigmoid')
])

model.summary(print_fn=logging.info)

# Save Model Architecture
os.makedirs(CONFIG['figures_folder'], exist_ok=True)
plot_model(model, to_file=os.path.join(CONFIG['figures_folder'], f"model-{CONFIG['model_id']}.png"), show_shapes=True)

# Compile Model
opt = tf.keras.optimizers.RMSprop(learning_rate=CONFIG['learning_rate'])
model.compile(
    loss='binary_crossentropy',
    optimizer=opt,
    metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=CONFIG['patience'], restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=CONFIG['patience'], min_lr=1e-4)
]

# Train Model
start_time = time.time()
history = model.fit(
    x_train_scaled,
    y_train_scaled,
    batch_size=CONFIG['batch_size'],
    epochs=CONFIG['epochs'],
    validation_data=(x_val_scaled, y_val_scaled),
    #class_weight=class_weights,
    callbacks=callbacks
)
end_time = time.time()
logging.info(f"Training completed in {end_time - start_time:.2f} seconds.")

# Save Training History
os.makedirs(CONFIG['output_folder'], exist_ok=True)
with open(os.path.join(CONFIG['output_folder'], f"history-{CONFIG['model_id']}.json"), 'w') as f:
    json.dump(history.history, f)

# Save Model
model.save(os.path.join(CONFIG['output_folder'], f"model-{CONFIG['model_id']}.keras"))

# Evaluate Model
evaluation_results = model.evaluate(x_test_scaled, y_test_scaled)
test_loss = evaluation_results[0]
test_accuracy = evaluation_results[1]

# Predictions
y_test_pred_proba = model.predict(x_test_scaled)
y_test_pred_class = (y_test_pred_proba > 0.5).astype(int)

# Metrics
accuracy = accuracy_score(y_test_scaled, y_test_pred_class)
roc_auc = roc_auc_score(y_test_scaled, y_test_pred_proba)
f1 = f1_score(y_test_scaled, y_test_pred_class, average='macro')

logging.info(f"Test data Accuracy: {accuracy}")
logging.info(f"Test data ROC AUC: {roc_auc}")
logging.info(f"Test data F1-Score: {f1}")

metrics = {
    'test_loss': test_loss,
    'test_accuracy': test_accuracy,
    'roc_auc': roc_auc,
    'f1_score': f1
}

with open(os.path.join(CONFIG['output_folder'], f"metrics-{CONFIG['model_id']}.json"), 'w') as f:
    json.dump(metrics, f)

# Plot Training and Validation Metrics
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(CONFIG['figures_folder'], f"accuracy-{CONFIG['model_id']}.png"))

plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(CONFIG['figures_folder'], f"loss-{CONFIG['model_id']}.png"))
