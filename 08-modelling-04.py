'''
MODELLING - GAUSSIAN PROCESSES
'''
import os
import joblib
import logging
import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Define the variational GP model class
class VariationalGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution)
        super(VariationalGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Configure logging
script_name = os.path.splitext(os.path.basename(__file__))[0]  # Get script name without extension
log_file = f"{script_name}.log"  # Create log file name

if os.path.exists(log_file):
    os.remove(log_file)  # Delete log file if it exists

logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting training script")

# Model identification
MODEL_ID = '04'
logging.info(f"Model ID: {MODEL_ID}")

# Specify the folder to export the figures
FIGURES_FOLDER = 'figures'
os.makedirs(FIGURES_FOLDER, exist_ok=True)  # Create folder if it does not exist
logging.info(f"Figures folder: {FIGURES_FOLDER}")

# Specify data and output folders
INPUT_FOLDER = 'input_files'
OUTPUT_FOLDER = 'output_files'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
logging.info(f"Input folder: {INPUT_FOLDER}")
logging.info(f"Output folder: {OUTPUT_FOLDER}")

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

# Convert data to torch tensors
x_train_torch = torch.tensor(x_train_scaled, dtype=torch.float32)
y_train_torch = torch.tensor(y_train_scaled, dtype=torch.float32)
x_val_torch = torch.tensor(x_val_scaled, dtype=torch.float32)
y_val_torch = torch.tensor(y_val_scaled, dtype=torch.float32)
x_test_torch = torch.tensor(x_test_scaled, dtype=torch.float32)
y_test_torch = torch.tensor(y_test_scaled, dtype=torch.float32)

# Adjust for single-output GP
num_outputs = y_train_torch.shape[1]

# Create a DataLoader for batching
batch_size = 1024  # Adjust based on memory capacity
train_dataset = TensorDataset(x_train_torch, y_train_torch)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Train separate variational models for each output
models = []
likelihoods = []
inducing_points = x_train_torch[:500]  # Use 500 inducing points
for i in range(num_outputs):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = VariationalGPModel(inducing_points)
    models.append(model)
    likelihoods.append(likelihood)

# Train each model independently with batching
for i, (model, likelihood) in enumerate(zip(models, likelihoods)):
    model.train()
    likelihood.train()
    
    # Use the Adam optimizer
    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=0.01)

    # Loss function (variational ELBO)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=x_train_torch.size(0))

    logging.info(f"Training model for output {i + 1}/{num_outputs}")
    num_iter = 50
    for epoch in range(num_iter):
        epoch_loss = 0.0
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_y = batch_y[:, i]  # Select the corresponding output dimension
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = -mll(output, batch_y)
            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item()
            epoch_loss += batch_loss
            print(f"Output {i + 1}, Epoch {epoch + 1}/{num_iter}, Batch {batch_idx + 1}/{len(train_loader)} - Batch Loss: {batch_loss:.4f}")

        logging.info(f"Output {i + 1}, Epoch {epoch + 1}/{num_iter} - Epoch Loss: {epoch_loss:.4f}")

    logging.info(f"Finished training model for output {i + 1}. Final Epoch Loss: {epoch_loss:.4f}")

# Evaluate models
y_val_pred = []
y_test_pred = []

for i, model in enumerate(models):
    model.eval()
    likelihoods[i].eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        y_val_pred.append(likelihoods[i](model(x_val_torch)).mean.numpy())
        y_test_pred.append(likelihoods[i](model(x_test_torch)).mean.numpy())

# Combine predictions
y_val_pred = np.column_stack(y_val_pred)
y_test_pred = np.column_stack(y_test_pred)

# Calculate R^2 scores
r2_val = r2_score(y_val_scaled, y_val_pred)
r2_test = r2_score(y_test_scaled, y_test_pred)
logging.info(f"R^2 score on validation set: {r2_val:.4f}")
logging.info(f"R^2 score on test set: {r2_test:.4f}")

# Plot true vs predicted for validation set
plt.figure()
plt.scatter(y_val_scaled, y_val_pred, alpha=0.6)
plt.plot([y_val_scaled.min(), y_val_scaled.max()], [y_val_scaled.min(), y_val_scaled.max()], 'k--', lw=2)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title(f"Validation Set: True vs Predicted (R^2={r2_val:.4f})")
plt.savefig(os.path.join(FIGURES_FOLDER, 'validation_true_vs_predicted.png'))
logging.info("Validation plot saved.")

# Plot true vs predicted for test set
plt.figure()
plt.scatter(y_test_scaled, y_test_pred, alpha=0.6)
plt.plot([y_test_scaled.min(), y_test_scaled.max()], [y_test_scaled.min(), y_test_scaled.max()], 'k--', lw=2)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title(f"Test Set: True vs Predicted (R^2={r2_test:.4f})")
plt.savefig(os.path.join(FIGURES_FOLDER, 'test_true_vs_predicted.png'))
logging.info("Test plot saved.")
