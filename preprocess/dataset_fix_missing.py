import cv2
import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from shutil import copy

dataset_input_path = "dataset"
dataset_output_path = "fix_dataset"

# Make path if not exist
Path(dataset_output_path).mkdir(parents=True, exist_ok=True)
Path(os.path.join(dataset_output_path, "images")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(dataset_output_path, "masks", "bad")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(dataset_output_path, "masks", "good")).mkdir(parents=True, exist_ok=True)

# Get file list
files_list = list()
for filename in os.listdir(os.path.join(dataset_input_path, "images")):
    files_list.append(os.path.splitext(filename)[0])

# Copy file to train and validation folder
for filename in files_list:
    copy(os.path.join(dataset_input_path, "images", filename + ".jpg"), os.path.join(dataset_output_path, "images", filename + ".jpg"))
    
    mask_bad_path = os.path.join(dataset_input_path, "masks", "bad", filename + ".png")
    if not os.path.exists(mask_bad_path):
        im = cv2.imread(os.path.join(dataset_input_path, "images", filename + ".jpg"))
        img = np.zeros((im.shape[0], im.shape[1], 3), dtype = "uint8")
        cv2.imwrite(os.path.join(dataset_output_path, "masks", "bad", filename + ".png"), img)
    else:
        copy(os.path.join(dataset_input_path, "masks", "bad", filename + ".png"), os.path.join(dataset_output_path, "masks", "bad", filename + ".png"))
    
    mask_good_path = os.path.join(dataset_input_path, "masks", "good", filename + ".png")
    if not os.path.exists(mask_good_path):
        im = cv2.imread(os.path.join(dataset_input_path, "images", filename + ".jpg"))
        img = np.zeros((im.shape[0], im.shape[1], 3), dtype = "uint8")
        cv2.imwrite(os.path.join(dataset_output_path, "masks", "good", filename + ".png"), img)
    else:
        copy(os.path.join(dataset_input_path, "masks", "good", filename + ".png"), os.path.join(dataset_output_path, "masks", "good", filename + ".png"))
