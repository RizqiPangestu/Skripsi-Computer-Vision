import cv2
import os
from pathlib import Path

image_size = (640, 480)
dataset_input_path = os.path.join("resized_dataset")
dataset_output_path = os.path.join("augment_dataset")

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
    cv2.imwrite(os.path.join(dataset_output_path, "images", filename + ".jpg"), image)
    if(image.shape == (640,640,3)):
        cv2.imwrite(os.path.join(dataset_output_path, "images", filename + "rotated.jpg"), cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE))
        cv2.imwrite(os.path.join(dataset_output_path, "images", filename + "rotated2.jpg"), cv2.rotate(image,cv2.ROTATE_180))
        
    # Load mask
    mask_0 = cv2.imread(os.path.join(dataset_input_path, "masks", "bad", filename + ".png"))
    if(mask_0 is not None):
        cv2.imwrite(os.path.join(dataset_output_path, "masks", "bad", filename + ".png"), mask_0)
        if(image.shape == (640,640,3)):
            cv2.imwrite(os.path.join(dataset_output_path, "masks", "bad", filename + "rotated.png"), cv2.rotate(mask_0,cv2.ROTATE_90_CLOCKWISE))
            cv2.imwrite(os.path.join(dataset_output_path, "masks", "bad", filename + "rotated2.png"), cv2.rotate(mask_0,cv2.ROTATE_180))

    mask_1 = cv2.imread(os.path.join(dataset_input_path, "masks", "good", filename + ".png"))
    if(mask_1 is not None):
        cv2.imwrite(os.path.join(dataset_output_path, "masks", "good", filename + ".png"), mask_1)
        if(image.shape == (640,640,3)):
            cv2.imwrite(os.path.join(dataset_output_path, "masks", "good", filename + "rotated.png"), cv2.rotate(mask_1,cv2.ROTATE_90_CLOCKWISE))
            cv2.imwrite(os.path.join(dataset_output_path, "masks", "good", filename + "rotated2.png"), cv2.rotate(mask_1,cv2.ROTATE_180))
