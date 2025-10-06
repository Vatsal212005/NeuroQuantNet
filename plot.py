import matplotlib.pyplot as plt
import os

# Create the plots directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Bar plot comparison
models = ['Baseline (Classical)', 'Hybrid (Old run)']
r2_scores = [0.786, 0.75]
mae_scores = [0.93, 0.95]
rmse_scores = [1.25, 1.30]

# Plot R2
plt.figure(figsize=(8, 5))
plt.bar(models, r2_scores)
plt.title("R² Score Comparison")
plt.ylim(0, 1)
plt.ylabel("R² Score")
plt.grid(True, axis='y')
plt.savefig("plots/r2_comparison.png")
plt.close()

# Plot MAE
plt.figure(figsize=(8, 5))
plt.bar(models, mae_scores)
plt.title("MAE Comparison")
plt.ylabel("MAE")
plt.grid(True, axis='y')
plt.savefig("plots/mae_comparison.png")
plt.close()

# Plot RMSE
plt.figure(figsize=(8, 5))
plt.bar(models, rmse_scores)
plt.title("RMSE Comparison")
plt.ylabel("RMSE")
plt.grid(True, axis='y')
plt.savefig("plots/rmse_comparison.png")
plt.close()

print("✅ Plots saved in the 'plots/' folder.")
