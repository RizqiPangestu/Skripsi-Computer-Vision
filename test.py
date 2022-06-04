import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def get_filelist(path):
        """
        Returns a list of absolute paths to images inside given `path`
        """
        files_list = list()
        for filename in os.listdir(path):
            files_list.append(os.path.splitext(filename)[0])
        return files_list

dataset_path = "dataset_old"

file_list = get_filelist(os.path.join(dataset_path, "images"))

# print(file_list)

kfold = KFold(5, True, 1)

for train, test in kfold.split(file_list):
    print(test)
    holder = [file_list[i] for i in test]
    print(holder)
    print("=================")