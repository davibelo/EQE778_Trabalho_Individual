'''
MODELLING - DEEP LEARNING
BINARY LABELS
'''
import os
import logging
import joblib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, regularizers
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

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
MODEL_ID = '07'

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

# Parameters
NEURONS_RATIO = 5  # Neurons per feature ratio
DROPOUT = 0.05       # Dropout rate

# Get number of input features and output targets from the training data
num_features = x_train_scaled.shape[1]  # Number of input features
num_outputs = y_train_scaled.shape[1]   # Number of output labels

# Function to determine the number of neurons in dense layers, rounded to the nearest multiple
def neurons(num_features, ratio, multiple=8):
    neuron_count = int(num_features * ratio)
    return max(multiple, round(neuron_count / multiple) * multiple)

# Define the model
model = tf.keras.Sequential()

model.add(layers.Input(shape=(num_features,)))

model.add(layers.Dense(neurons(num_features, NEURONS_RATIO),                        
                       kernel_regularizer=regularizers.l2(0.0001)))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())
model.add(layers.Dropout(0.3))

model.add(layers.Dense(neurons(num_features, NEURONS_RATIO / 2), 
                       kernel_regularizer=regularizers.l2(0.0001)))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())
model.add(layers.Dropout(0.2))

model.add(layers.Dense(neurons(num_features, NEURONS_RATIO / 4),                        
                       kernel_regularizer=regularizers.l2(0.0001)))
model.add(layers.ReLU())
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.1))

model.add(layers.Dense(num_outputs, activation='sigmoid'))

# Display the model summary
model.summary(print_fn=logging.info)

# Define the optimizer and compile the model
#opt = tf.keras.optimizers.Adam(learning_rate=0.001)
opt = tf.keras.optimizers.RMSprop(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# Define early stopping criteria
early_stopping = tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)

# Add additional callbacks if needed (e.g., ReduceLROnPlateau)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                 factor=0.2,
                                                 patience=20, 
                                                 min_lr=0.0001)

# Train the model
history = model.fit(x_train_scaled, 
                    y_train_scaled, 
                    batch_size=128, 
                    epochs=100, 
                    validation_data=(x_val_scaled, y_val_scaled), 
                    callbacks=[early_stopping, reduce_lr],
                    verbose=1)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(x_test_scaled, y_test_scaled)
logging.info(f"Test loss: {test_loss}")
logging.info(f"Test accuracy: {test_accuracy}")

# Save the model in Keras native format
model.save(os.path.join(OUTPUT_FOLDER, f'model-{MODEL_ID}.keras'))

# Load the model
loaded_model = tf.keras.models.load_model(os.path.join(OUTPUT_FOLDER, f'model-{MODEL_ID}.keras'))

# Predict probabilities and binary classifications
y_test_pred_proba = model.predict(x_test_scaled)
y_test_pred_class = (y_test_pred_proba > 0.5).astype(int)

# Calculate additional metrics
accuracy = accuracy_score(y_test_scaled, y_test_pred_class)
roc_auc = roc_auc_score(y_test_scaled, y_test_pred_proba)
logging.info(f"Test data Accuracy: {accuracy}")
logging.info(f"Test data ROC AUC: {roc_auc}")

