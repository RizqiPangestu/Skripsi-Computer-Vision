import cv2
import os
from pathlib import Path
from shutil import copy

dataset_input_path = "augmented_dataset"
dataset_output_path = "sobel_dataset"

# Preprocess
def sobel(img):
    gx = cv2.Sobel(image,cv2.CV_64F,dx=1,dy=0)
    gy = cv2.Sobel(image,cv2.CV_64F,dx=0,dy=1)
    gx = cv2.convertScaleAbs(gx)
    gy = cv2.convertScaleAbs(gy)
    sobel = cv2.addWeighted(gx,0.5,gy,0.5,0)
    return sobel

preprocess = sobel

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
