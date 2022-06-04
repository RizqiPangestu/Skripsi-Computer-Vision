import cv2
import os
from pathlib import Path

image_size = (640, 480)
dataset_input_path = os.path.join("dataset")
dataset_output_path = os.path.join("resized_dataset")

# Make path if not exist
Path(dataset_output_path).mkdir(parents=True, exist_ok=True)
Path(os.path.join(dataset_output_path, "images")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(dataset_output_path, "masks", "bad")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(dataset_output_path, "masks", "good")).mkdir(parents=True, exist_ok=True)

files_list = list()
for filename in os.listdir(os.path.join(dataset_input_path, "images")):
    files_list.append(os.path.splitext(filename)[0])

for filename in files_list:
    print(filename)

    # Load image
    image = cv2.imread(os.path.join(dataset_input_path, "images", filename + ".jpg"))
    print(image.shape)
    if(image.shape == (3000,3000,3)):
        image_size = (640,640)
    else:
        image_size = (640,480)
    
    image = cv2.resize(image, image_size)
    cv2.imwrite(os.path.join(dataset_output_path, "images", filename + ".jpg"), image)
    
    # Load mask
    mask_0 = cv2.imread(os.path.join(dataset_input_path, "masks", "bad", filename + ".png"))
    if(mask_0 is not None):
        mask_0 = cv2.resize(mask_0, image_size, interpolation = cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(dataset_output_path, "masks", "bad", filename + ".png"), mask_0)

    mask_1 = cv2.imread(os.path.join(dataset_input_path, "masks", "good", filename + ".png"))
    if(mask_1 is not None):
        mask_1 = cv2.resize(mask_1, image_size, interpolation = cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(dataset_output_path, "masks", "good", filename + ".png"), mask_1)
