import cv2
import os
from pathlib import Path
from shutil import copy
from skimage import feature

dataset_input_path = "augmented_dataset"
dataset_output_path = "lbp_dataset"

# Preprocess
def lbp(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(gray,44,2, method="ror")
    return lbp

preprocess = lbp

# Make path if not exist
Path(dataset_output_path).mkdir(parents=True, exist_ok=True)
Path(os.path.join(dataset_output_path, "train", "images")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(dataset_output_path, "train", "masks", "bad")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(dataset_output_path, "train", "masks", "good")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(dataset_output_path, "valid", "images")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(dataset_output_path, "valid", "masks", "bad")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(dataset_output_path, "valid", "masks", "good")).mkdir(parents=True, exist_ok=True)

# Get train file list
train_list = list()
for filename in os.listdir(os.path.join(dataset_input_path, "train", "images")):
    train_list.append(os.path.splitext(filename)[0])

# Preprocess Train
for filename in train_list:
    image = cv2.imread(os.path.join(dataset_input_path, "train", "images", filename + ".jpg"))

    preprocess_image = preprocess(image)

    cv2.imwrite(os.path.join(dataset_output_path, "train", "images", filename + ".jpg"), preprocess_image)

    copy(os.path.join(dataset_input_path, "train", "masks", "bad", filename + ".png"), os.path.join(dataset_output_path, "train", "masks", "bad", filename + ".png"))
    copy(os.path.join(dataset_input_path, "train", "masks", "good", filename + ".png"), os.path.join(dataset_output_path, "train", "masks", "good", filename + ".png"))

# Get valid file list
valid_list = list()
for filename in os.listdir(os.path.join(dataset_input_path, "valid", "images")):
    valid_list.append(os.path.splitext(filename)[0])

# Preprocess valid
for filename in valid_list:
    image = cv2.imread(os.path.join(dataset_input_path, "valid", "images", filename + ".jpg"))
    
    preprocess_image = preprocess(image)

    cv2.imwrite(os.path.join(dataset_output_path, "valid", "images", filename + ".jpg"), preprocess_image)

    copy(os.path.join(dataset_input_path, "valid", "masks", "bad", filename + ".png"), os.path.join(dataset_output_path, "valid", "masks", "bad", filename + ".png"))
    copy(os.path.join(dataset_input_path, "valid", "masks", "good", filename + ".png"), os.path.join(dataset_output_path, "valid", "masks", "good", filename + ".png"))
