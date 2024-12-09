import os
import numpy as np
import matplotlib.pyplot as plt

# Data from the images
models = ['RF', 'DNN']
training_time = [1124, 15777]  # in seconds
metrics = {
    'cH2S_accuracy': [0.7420, 0.8357],
    'cH2S_roc_auc': [0.8314, 0.8185],
    'cH2S_f1_score': [0.7668, 0.7897],
    'cNH3_accuracy': [0.7575, 0.7711],
    'cNH3_roc_auc': [0.8459, 0.8365],
    'cNH3_f1_score': [0.7630, 0.7661]
}


# Create 'figures' directory if it doesn't exist
os.makedirs("figures", exist_ok=True)

# Save execution time comparison plot
plt.figure(figsize=(10, 6))
plt.bar(models, training_time, color=['blue', 'green'])
plt.title("Model Training Time Comparison", fontsize=14)
plt.ylabel("Training Time (seconds)", fontsize=12)
plt.xlabel("Model", fontsize=12)
plt.savefig("figures/training_time_comparison.png")
plt.close()

# Save cH2S metrics comparison plot
x = np.arange(len(models))
width = 0.2

plt.figure(figsize=(12, 8))
plt.bar(x - width, metrics['cH2S_accuracy'], width, label='Accuracy')
plt.bar(x, metrics['cH2S_roc_auc'], width, label='ROC AUC')
plt.bar(x + width, metrics['cH2S_f1_score'], width, label='F1 Score')
plt.xticks(x, models)
plt.title("cH2S Metrics Comparison", fontsize=14)
plt.ylabel("Metric Value", fontsize=12)
plt.xlabel("Model", fontsize=12)
plt.legend()
plt.savefig("figures/cH2S_metrics_comparison.png")
plt.close()

# Save cNH3 metrics comparison plot
plt.figure(figsize=(12, 8))
plt.bar(x - width, metrics['cNH3_accuracy'], width, label='Accuracy')
plt.bar(x, metrics['cNH3_roc_auc'], width, label='ROC AUC')
plt.bar(x + width, metrics['cNH3_f1_score'], width, label='F1 Score')
plt.xticks(x, models)
plt.title("cNH3 Metrics Comparison", fontsize=14)
plt.ylabel("Metric Value", fontsize=12)
plt.xlabel("Model", fontsize=12)
plt.legend()
plt.savefig("figures/cNH3_metrics_comparison.png")
plt.close()
