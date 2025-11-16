import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
base_dir = "Faulty_solar_panel"  # your current dataset
output_dir = "dataset/images"

# Create output folders
for folder in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

# Parameters
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Go through each class folder (Bird-drop, Dusty, etc.)
for class_name in os.listdir(base_dir):
    class_path = os.path.join(base_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    train_files, temp_files = train_test_split(images, train_size=train_ratio, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=test_ratio/(test_ratio+val_ratio), random_state=42)

    # Copy files to new folders
    for file_list, split_name in zip([train_files, val_files, test_files], ['train', 'val', 'test']):
        for f in file_list:
            src = os.path.join(class_path, f)
            dst = os.path.join(output_dir, split_name, f)
            shutil.copy2(src, dst)

print("Dataset split completed successfully!")
