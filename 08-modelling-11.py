'''
MODELLING - MCTB-DRSN trial
'''
import os
import logging
import joblib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, regularizers, models
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
MODEL_ID = '11'

# Specify the folder to export the figures
FIGURES_FOLDER = 'figures'

# Specify data and output folders
INPUT_FOLDER = 'input_files'
OUTPUT_FOLDER = 'output_files'

# Import x and y dataframes
df_scaled_x = joblib.load(os.path.join(INPUT_FOLDER, 'df3_scaled_x.joblib'))
df_scaled_y = joblib.load(os.path.join(INPUT_FOLDER, 'df3_scaled_y.joblib'))

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

# Reshape data for CNN input (assuming num_features is a perfect square)
num_features = x_train_scaled.shape[1]
H = int(np.sqrt(num_features))
W = H
assert H * W == num_features, "Number of features must be a perfect square for reshaping into 2D grid."

x_train_reshaped = x_train_scaled.reshape(-1, H, W, 1)
x_val_reshaped = x_val_scaled.reshape(-1, H, W, 1)
x_test_reshaped = x_test_scaled.reshape(-1, H, W, 1)
x_all_reshaped = x_scaled.reshape(-1, H, W, 1)

logging.info(f"Reshaped x_train shape: {x_train_reshaped.shape}")
logging.info(f"Reshaped x_val shape: {x_val_reshaped.shape}")
logging.info(f"Reshaped x_test shape: {x_test_reshaped.shape}")

# Define Masked Convolution Layer
def masked_conv2d(inputs, filters, kernel_size):
    mask = tf.Variable(tf.ones((kernel_size, kernel_size, inputs.shape[-1], filters)), trainable=True)
    x = layers.Conv2D(filters, kernel_size, padding='same', activation=None)(inputs)
    return layers.Multiply()([x, mask])

# Define Global Response Normalization (GRN) Layer
def grn_layer(inputs):
    squared_sum = tf.reduce_sum(tf.square(inputs), axis=-1, keepdims=True)
    norm_factor = tf.sqrt(squared_sum + 1e-6)
    return layers.Lambda(lambda x: x / norm_factor)(inputs)

# Define Residual Block with Shrinkage Mechanism
def res_block(inputs):
    x = layers.BatchNormalization()(inputs)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = grn_layer(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    
    abs_x = tf.abs(x)
    gap = layers.GlobalAveragePooling2D()(abs_x)
    dense = layers.Dense(64, activation='relu')(gap)
    dense = layers.Dense(64, activation='sigmoid')(dense)
    dense = tf.expand_dims(tf.expand_dims(dense, 1), 1)  # Reshape for multiplication
    scale = layers.Multiply()([x, dense])
    
    x = layers.Subtract()([x, scale])
    x = layers.Maximum()([x, scale])
    x = layers.Multiply()([x, scale])
    
    return x

# Define MCTB Block with Channel-wise Transformer Attention
def mctb_block(inputs):
    x = masked_conv2d(inputs, 64, 3)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    
    attn_weights = layers.Dense(64, activation='softmax')(x)
    x = layers.Multiply()([x, attn_weights])
    x = layers.LayerNormalization()(x)
    
    ff = layers.Dense(64, activation='relu')(x)
    x = layers.Add()([x, ff])
    x = layers.LayerNormalization()(x)
    
    x = layers.Multiply()([x, inputs])
    return x

# Define the full model
def build_model(input_shape, num_outputs):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = mctb_block(x)
    x = res_block(x)
    x = res_block(x)
    x = layers.Dropout(0.5)(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_outputs, activation='linear')(x)
    
    model = models.Model(inputs, outputs)
    return model

# Build the model
num_outputs = y_train_scaled.shape[1]
model = build_model(input_shape=(H, W, 1), num_outputs=num_outputs)
model.summary(print_fn=logging.info)

# Define the optimizer and compile the model
opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(loss='mse', optimizer=opt, metrics=['mae'])

# Define early stopping criteria
early_stopping = tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, min_lr=0.0001)

# Train the model
history = model.fit(x_train_reshaped, 
                    y_train_scaled, 
                    batch_size=128, 
                    epochs=100, 
                    validation_data=(x_val_reshaped, y_val_scaled), 
                    callbacks=[early_stopping, reduce_lr],
                    verbose=1)

# Evaluate the model on the test data
test_loss = model.evaluate(x_test_reshaped, y_test_scaled)
logging.info(f"Test loss: {test_loss}")

# Save the model in Keras native format
model.save(os.path.join(OUTPUT_FOLDER, f'model-{MODEL_ID}.keras'))

# Load the model
loaded_model = tf.keras.models.load_model(os.path.join(OUTPUT_FOLDER, f'model-{MODEL_ID}.keras'))

# Generate predictions
y_test_pred_scaled = model.predict(x_test_reshaped)
y_pred_scaled = model.predict(x_all_reshaped)

# Calculate residues
residue_test = y_test_scaled.ravel() - y_test_pred_scaled.ravel()
residue = y_scaled.ravel() - y_pred_scaled.ravel()

# Calculate RMSE
rmse_dl = np.sqrt(model.evaluate(x_test_reshaped, y_test_scaled, batch_size=1000, verbose=0)[0])
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
