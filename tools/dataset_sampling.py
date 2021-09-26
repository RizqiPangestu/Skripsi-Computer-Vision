import os
import random
import shutil
from pathlib import Path

sample_size = 500
dataset_input_path = os.path.join("dataset")
dataset_output_path = os.path.join("sampled_dataset_" + str(sample_size))

# Make path if not exist
Path(dataset_output_path).mkdir(parents=True, exist_ok=True)
Path(os.path.join(dataset_output_path, "images")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(dataset_output_path, "masks", "background")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(dataset_output_path, "masks", "0")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(dataset_output_path, "masks", "1")).mkdir(parents=True, exist_ok=True)

files_list = list()
for filename in os.listdir(os.path.join(dataset_input_path, "images")):
    files_list.append(os.path.splitext(filename)[0])

sampled_files_list = random.sample(files_list, sample_size)

for filename in sampled_files_list:
    print(filename)
    # Copy Image
    shutil.copy(os.path.join(dataset_input_path, "images", filename + ".jpg"), os.path.join(dataset_output_path, "images", filename + ".jpg"))
    # Copy Masks
    shutil.copy(os.path.join(dataset_input_path, "masks", "background", filename + ".png"), os.path.join(dataset_output_path, "masks", "background", filename + ".png"))
    shutil.copy(os.path.join(dataset_input_path, "masks", "0", filename + ".png"), os.path.join(dataset_output_path, "masks", "0", filename + ".png"))
    shutil.copy(os.path.join(dataset_input_path, "masks", "1", filename + ".png"), os.path.join(dataset_output_path, "masks", "1", filename + ".png"))

print(sample_size, "files processed")
