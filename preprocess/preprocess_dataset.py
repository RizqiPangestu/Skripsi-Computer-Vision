import cv2
import os
from pathlib import Path
from skimage import feature
import argparse

parser = argparse.ArgumentParser(
    prog="preprocess_dataset.py", description="Preprocessing Dataset"
)
parser.add_argument(
    '--method', action='store', default='hsv', type=str, dest='method'
)

args = parser.parse_args()
method = args.method


def hsv(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    h,s,v = cv2.split(hsv)
    return hsv

def gray(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return gray

def sobel(img):
    gx = cv2.Sobel(image,cv2.CV_64F,dx=1,dy=0)
    gy = cv2.Sobel(image,cv2.CV_64F,dx=0,dy=1)
    gx = cv2.convertScaleAbs(gx)
    gy = cv2.convertScaleAbs(gy)
    sobel = cv2.addWeighted(gx,0.5,gy,0.5,0)
    return sobel

def lbp(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(gray,44,2, method="ror")
    return lbp

if method == 'hsv':
    preprocess = hsv
elif method == 'gray':
    preprocess = gray
elif method == 'sobel':
    preprocess = sobel
elif method == 'lbp':
    preprocess = lbp

dataset_input_path = os.path.join("dataset")
dataset_output_path = os.path.join("dataset_" + method)

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
    cv2.imwrite(os.path.join(dataset_output_path, "images", filename + ".jpg"), preprocess(image))
    
    # Load mask
    mask_0 = cv2.imread(os.path.join(dataset_input_path, "masks", "bad", filename + ".png"))
    if(mask_0 is not None):
        cv2.imwrite(os.path.join(dataset_output_path, "masks", "bad", filename + ".png"), mask_0)

    mask_1 = cv2.imread(os.path.join(dataset_input_path, "masks", "good", filename + ".png"))
    if(mask_1 is not None):
        cv2.imwrite(os.path.join(dataset_output_path, "masks", "good", filename + ".png"), mask_1)
