import cv2
import numpy as np
from numpy.lib.index_tricks import AxisConcatenator

image1 = cv2.imread('dataset_lbp/images/2021-07-08-135743g1b1.jpg',cv2.IMREAD_GRAYSCALE)
image1 = np.expand_dims(image1,axis=2)
image2 = cv2.imread('dataset/images/2021-07-08-135743g1b1.jpg')
print(image1.shape)
print(image2.shape)
image = np.concatenate((image1,image2))
print(image.shape)