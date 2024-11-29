'''
MODELLING - DEEP LEARNING
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
from sklearn.metrics import r2_score

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
MODEL_ID = '05'

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

# Input layer with the correct shape based on x_train_scaled
model.add(layers.Input(shape=(num_features,)))

# Dense layers with Dropout, Batch Normalization, and L2 regularization
model.add(layers.Dense(neurons(num_features, NEURONS_RATIO), 
                       activation='softplus', 
                       kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dropout(DROPOUT))
model.add(layers.BatchNormalization())

model.add(layers.Dense(neurons(num_features, NEURONS_RATIO / 2), 
                       activation='softplus', 
                       kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dropout(DROPOUT / 2))
model.add(layers.BatchNormalization())

model.add(layers.Dense(neurons(num_features, NEURONS_RATIO / 4), 
                       activation='softplus', 
                       kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dropout(DROPOUT / 4))
model.add(layers.BatchNormalization())

# Output layer with neurons matching the number of outputs in y_train_scaled
model.add(layers.Dense(num_outputs))

# Display the model summary
model.summary(print_fn=logging.info)

# Define the optimizer and compile the model
opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(loss='mse', optimizer=opt)

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
test_loss = model.evaluate(x_test_scaled, y_test_scaled)
logging.info(f"Test loss: {test_loss}")

# Save the model in Keras native format
model.save(os.path.join(OUTPUT_FOLDER, f'model-{MODEL_ID}.keras'))

# Load the model
loaded_model = tf.keras.models.load_model(os.path.join(OUTPUT_FOLDER, f'model-{MODEL_ID}.keras'))

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
plt.savefig(f'{FIGURES_FOLDER}/model_{MODEL_ID}_test_residue.png')

plt.close()

