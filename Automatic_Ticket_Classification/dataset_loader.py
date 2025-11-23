import pandas as pd
from datasets import load_dataset, DatasetDict 

print("Downloading dataset from Hugging Face...")

# Load the original dataset (only 'train' exists in HF)
dataset = load_dataset("Tobi-Bueck/customer-support-tickets")
full_train = dataset["train"]

print("Original Dataset Loaded!")
print("Shape:", full_train.shape)

# Split: 80% train, 20% temp (validation+test)
print("\n Splitting dataset into Train 80% and Temp 20%...")
split1 = full_train.train_test_split(test_size=0.2, seed=42)

# Split Temp into 10% validation + 10% test
print("Splitting Temp into Validation 10% and Test 10%...")
temp = split1["test"]
split2 = temp.train_test_split(test_size=0.5, seed=42)

# Combine into DatasetDict
final_ds = DatasetDict({
    "train": split1["train"],
    "validation": split2["train"],
    "test": split2["test"]
})

print("\n Final Split Sizes:")
for split in final_ds:
    print(f"{split}: {len(final_ds[split])} rows")

# Save each split to CSV
print("\n Saving CSV files...")

final_ds["train"].to_pandas().to_csv("tickets_train.csv", index=False)
final_ds["validation"].to_pandas().to_csv("tickets_validation.csv", index=False)
final_ds["test"].to_pandas().to_csv("tickets_test.csv", index=False)

print("CSV files saved:")
print("    tickets_train.csv")
print("    tickets_validation.csv")
print("    tickets_test.csv")

#  Show small sample
print("\n Sample Rows:")
print(final_ds["train"].to_pandas().head())
