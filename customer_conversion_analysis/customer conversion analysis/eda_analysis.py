import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure a folder exists for charts
os.makedirs("plots", exist_ok=True)

# Load the original training data (not the transformed .pkl)
data = pd.read_csv(r"C:\Users\91902\OneDrive\Desktop\customer conversion analysis\train_data.csv")

# Quick overview
print("\n--- Basic Info ---")
print(data.info())
print("\n--- Summary Stats ---")
print(data.describe())

# Missing values
print("\n--- Missing Values ---")
print(data.isnull().sum())

# Distribution of each numeric column
num_cols = data.select_dtypes(exclude="object").columns
for col in num_cols:
    plt.figure(figsize=(6,3))
    sns.histplot(data[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(f"plots/dist_{col}.png")
    plt.close()

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(data[num_cols].corr(), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png")
plt.close()

print("EDA complete. Plots saved to 'plots' folder.")
