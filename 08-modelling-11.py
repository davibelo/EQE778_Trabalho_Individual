'''
MODELLING - MCTB-DRSN trial - 1D Adaptation
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

# Modified Masked Convolution for 1D
@tf.keras.utils.register_keras_serializable()
class MaskedConv1D(layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(MaskedConv1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv = layers.Conv1D(filters, kernel_size, padding='same', activation=None)
        self.mask = tf.Variable(tf.ones((1, 1, filters)), trainable=True)  # Channel-wise scaling

    def build(self, input_shape):
        self.conv.build(input_shape)
        super(MaskedConv1D, self).build(input_shape)

    def call(self, inputs):
        x = self.conv(inputs)
        mask = tf.broadcast_to(self.mask, tf.shape(x))
        return layers.Multiply()([x, mask])

# Keep GRN Layer unchanged (works with any dimensionality)
@tf.keras.utils.register_keras_serializable()
class GRNLayer(layers.Layer):
    def call(self, inputs):
        squared_sum = tf.reduce_sum(tf.square(inputs), axis=-1, keepdims=True)
        norm_factor = tf.sqrt(squared_sum + 1e-6)
        return inputs / norm_factor

# Modified Residual Block for 1D
@tf.keras.utils.register_keras_serializable()
class ResidualBlock1D(layers.Layer):
    def __init__(self, **kwargs):
        super(ResidualBlock1D, self).__init__(**kwargs)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.conv1 = layers.Conv1D(64, 3, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.grn = GRNLayer()
        self.conv2 = layers.Conv1D(64, 3, padding='same')
        self.gap = layers.GlobalAveragePooling1D()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(64, activation='sigmoid')

    def build(self, input_shape):
        self.conv1.build(input_shape)
        self.conv2.build(input_shape)
        super(ResidualBlock1D, self).build(input_shape)

    def call(self, inputs):
        x = self.bn1(inputs)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.grn(x)
        x = self.conv2(x)
        
        abs_x = tf.abs(x)
        gap = self.gap(abs_x)
        dense = self.dense1(gap)
        dense = self.dense2(dense)
        dense = tf.expand_dims(dense, 1)  # Add sequence dimension
        return layers.Multiply()([x, dense])

# Modified MCTB Block for 1D
@tf.keras.utils.register_keras_serializable()
class MCTBBlock1D(layers.Layer):
    def __init__(self, **kwargs):
        super(MCTBBlock1D, self).__init__(**kwargs)
        self.masked_conv = MaskedConv1D(64, 3)
        self.gap = layers.GlobalAveragePooling1D()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(64, activation='softmax')
        self.layer_norm1 = layers.LayerNormalization()
        self.ff = layers.Dense(64, activation='relu')
        self.layer_norm2 = layers.LayerNormalization()

    def build(self, input_shape):
        self.masked_conv.build(input_shape)
        super(MCTBBlock1D, self).build(input_shape)

    def call(self, inputs):
        x = self.masked_conv(inputs)
        x = self.gap(x)
        x = self.dense1(x)
        
        attn_weights = self.dense2(x)
        x = layers.Multiply()([x, attn_weights])
        x = self.layer_norm1(x)
        
        ff = self.ff(x)
        x = layers.Add()([x, ff])
        x = self.layer_norm2(x)
        
        return layers.Multiply()([x, inputs])

# Modified Model Builder
def build_model(input_shape, num_outputs):
    inputs = layers.Input(shape=input_shape)
    
    # Feature extraction
    x = MaskedConv1D(64, 3)(inputs)
    x = MCTBBlock1D()(x)
    x = ResidualBlock1D()(x)
    
    # Prediction head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_outputs, activation='linear')(x)
    
    return models.Model(inputs, outputs)

# Data reshaping for 1D Conv
x_train_reshaped = x_train_scaled.reshape(-1, 5, 1)
x_val_reshaped = x_val_scaled.reshape(-1, 5, 1)
x_test_reshaped = x_test_scaled.reshape(-1, 5, 1)

# Build and compile model
model = build_model((5, 1), 1)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

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

# Ensure the shapes match before calculating residues
y_test_scaled = y_test_scaled[:len(y_test_pred_scaled)]

# Calculate residues
residue_test = y_test_scaled.ravel() - y_test_pred_scaled.ravel()

# Define x_all_reshaped
x_all_reshaped = x_scaled.reshape(-1, 5, 1)

y_pred_scaled = model.predict(x_all_reshaped)

# Calculate residues
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
