import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
def build_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = mctb_block(x)
    x = res_block(x)
    x = res_block(x)
    x = layers.Dropout(0.5)(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(1, activation='linear')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Load dataset (assuming CSV format for Pandas)
def load_and_preprocess_data(csv_file):
    df = pd.read_csv(csv_file)
    X = df.iloc[:, :-1].values  # Features
    y = df.iloc[:, -1].values   # Target
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X = X.reshape((-1, 8, 8, 1))  # Reshape for CNN input (modify as needed)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Training function
def train_model(csv_file):
    X_train, X_test, y_train, y_test = load_and_preprocess_data(csv_file)
    model = build_model(input_shape=(8, 8, 1))  # Adjust input shape accordingly
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    model.save('mctb_drsn_model.h5')
    return model

# Example usage
# model = train_model('dataset.csv')
