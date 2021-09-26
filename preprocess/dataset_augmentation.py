import cv2
import os
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from shutil import copy
import albumentations as A

random_seed = 4269
augmentation_pass = 5
dataset_input_path = "split_dataset"
dataset_output_path = "augmented_dataset"

random.seed(random_seed)

# Augmentation
transform = A.Compose([
    A.RandomRotate90(),
    A.Flip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.GaussNoise(),
    A.HueSaturationValue(),
    A.RandomBrightnessContrast(),
    A.Rotate(p=1.0, limit=(-90, 90), interpolation=0, border_mode=4),
    A.RandomCrop(480, 480),
], additional_targets={'mask0': 'mask'})

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

# Augment Train
for filename in train_list:
    image = cv2.imread(os.path.join(dataset_input_path, "train", "images", filename + ".jpg"))
    # Load mask
    mask_0 = cv2.imread(os.path.join(dataset_input_path, "train", "masks", "bad", filename + ".png"), cv2.IMREAD_GRAYSCALE)
    mask_1 = cv2.imread(os.path.join(dataset_input_path, "train", "masks", "good", filename + ".png"), cv2.IMREAD_GRAYSCALE)

    for i in range(0, augmentation_pass):
        augmented = transform(image=image, mask=mask_0, mask0=mask_1)
        aug_image = augmented["image"]
        aug_mask_0 = augmented["mask"]
        aug_mask_1 = augmented["mask0"]

        cv2.imwrite(os.path.join(dataset_output_path, "train", "images", filename + "_" + str(i) + ".jpg"), aug_image)
        cv2.imwrite(os.path.join(dataset_output_path, "train", "masks", "bad", filename + "_" + str(i) + ".png"), aug_mask_0)
        cv2.imwrite(os.path.join(dataset_output_path, "train", "masks", "good", filename + "_" + str(i) + ".png"), aug_mask_1)

# Get valid file list
valid_list = list()
for filename in os.listdir(os.path.join(dataset_input_path, "valid", "images")):
    valid_list.append(os.path.splitext(filename)[0])

# Augment valid
for filename in valid_list:
    image = cv2.imread(os.path.join(dataset_input_path, "valid", "images", filename + ".jpg"))
    # Load mask
    mask_0 = cv2.imread(os.path.join(dataset_input_path, "valid", "masks", "bad", filename + ".png"), cv2.IMREAD_GRAYSCALE)
    mask_1 = cv2.imread(os.path.join(dataset_input_path, "valid", "masks", "good", filename + ".png"), cv2.IMREAD_GRAYSCALE)

    for i in range(0, augmentation_pass):
        augmented = transform(image=image, mask=mask_0, mask0=mask_1)
        aug_image = augmented["image"]
        aug_mask_0 = augmented["mask"]
        aug_mask_1 = augmented["mask0"]

        cv2.imwrite(os.path.join(dataset_output_path, "valid", "images", filename + "_" + str(i) + ".jpg"), aug_image)
        cv2.imwrite(os.path.join(dataset_output_path, "valid", "masks", "bad", filename + "_" + str(i) + ".png"), aug_mask_0)
        cv2.imwrite(os.path.join(dataset_output_path, "valid", "masks", "good", filename + "_" + str(i) + ".png"), aug_mask_1)
