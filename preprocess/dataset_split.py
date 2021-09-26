import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from shutil import copy

val_size = 0.2
random_seed = 4269
dataset_input_path = "fix_dataset"
dataset_output_path = "split_dataset"

# Make path if not exist
Path(dataset_output_path).mkdir(parents=True, exist_ok=True)
Path(os.path.join(dataset_output_path, "train", "images")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(dataset_output_path, "train", "masks", "bad")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(dataset_output_path, "train", "masks", "good")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(dataset_output_path, "valid", "images")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(dataset_output_path, "valid", "masks", "bad")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(dataset_output_path, "valid", "masks", "good")).mkdir(parents=True, exist_ok=True)

# Get file list
files_list = list()
for filename in os.listdir(os.path.join(dataset_input_path, "images")):
    files_list.append(os.path.splitext(filename)[0])

# Split to train and validation
train_list, valid_list = train_test_split(files_list, shuffle=True, test_size=val_size, random_state=random_seed)

# Copy file to train and validation folder
for filename in train_list:
    copy(os.path.join(dataset_input_path, "images", filename + ".jpg"), os.path.join(dataset_output_path, "train", "images", filename + ".jpg"))
    copy(os.path.join(dataset_input_path, "masks", "bad", filename + ".png"), os.path.join(dataset_output_path, "train", "masks", "bad", filename + ".png"))
    copy(os.path.join(dataset_input_path, "masks", "good", filename + ".png"), os.path.join(dataset_output_path, "train", "masks", "good", filename + ".png"))

for filename in valid_list:
    copy(os.path.join(dataset_input_path, "images", filename + ".jpg"), os.path.join(dataset_output_path, "valid", "images", filename + ".jpg"))
    copy(os.path.join(dataset_input_path, "masks", "bad", filename + ".png"), os.path.join(dataset_output_path, "valid", "masks", "bad", filename + ".png"))
    copy(os.path.join(dataset_input_path, "masks", "good", filename + ".png"), os.path.join(dataset_output_path, "valid", "masks", "good", filename + ".png"))
