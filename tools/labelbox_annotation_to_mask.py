import cv2
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from time import time as timer

# Images path
images_output_path = os.path.join("images")
# Annotation path
annotation_path = os.path.join("annotation", "labelbox_export_csv.csv")
# Output path
masks_output_path = os.path.join("masks")

# Make path if not exist
Path(images_output_path).mkdir(parents=True, exist_ok=True)
Path(masks_output_path).mkdir(parents=True, exist_ok=True)
Path(os.path.join(masks_output_path, "background")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(masks_output_path, "0")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(masks_output_path, "1")).mkdir(parents=True, exist_ok=True)

# Open CSV
data = pd.read_csv(annotation_path)

start = timer()
dataset_size = len(data)
for i in range(0, dataset_size):
    filename = os.path.splitext(data.iloc[i]["External ID"])[0]
    label = json.loads(data.iloc[i]["Label"])
    if label != {}: # Check if label empty
        print(filename)

        # Check if mask exist. if not create blank image
        mask_0_path = os.path.join(masks_output_path, "0", filename + ".png")
        if not os.path.exists(mask_0_path):
            im = cv2.imread(os.path.join(images_output_path, filename + ".jpg"))
            img = np.zeros((im.shape[0], im.shape[1], 3), dtype = "uint8")
            cv2.imwrite(mask_0_path, img)
        
        mask_1_path = os.path.join(masks_output_path, "1", filename + ".png")
        if not os.path.exists(mask_1_path):
            im = cv2.imread(os.path.join(images_output_path, filename + ".jpg"))
            img = np.zeros((im.shape[0], im.shape[1], 3), dtype = "uint8")
            cv2.imwrite(mask_1_path, img)

        # Draw each polygon
        segmentation_label_size = len(label["objects"])
        for j in range(0, segmentation_label_size):
            segmentation_label_value = label["objects"][j]["value"]

            # Load Polygon data
            poly = label["objects"][j]["polygon"]
            polygons = []
            for k in range(len(poly)):
                polygons.append([poly[k]['x'], poly[k]['y']])
            polygons = np.array(polygons).astype(int)

            # Draw Polygon
            mask_path = os.path.join(masks_output_path, segmentation_label_value, filename + ".png")
            mask = cv2.imread(mask_path)
            mask = cv2.fillPoly(mask, [polygons], (255, 255, 255))
            cv2.imwrite(mask_path, mask)

        # Generate Synthetic Label For Background
        mask_0 = cv2.imread(mask_0_path)
        mask_1 = cv2.imread(mask_1_path)
        mask_background = cv2.bitwise_or(mask_0, mask_1)
        mask_background = cv2.bitwise_not(mask_background)

        cv2.imwrite(os.path.join(masks_output_path, "background", filename + ".png"), mask_background)
    else:
        print("Missing label", filename, "skippping")

print("Elapsed Time:", timer() - start, "Seconds")
