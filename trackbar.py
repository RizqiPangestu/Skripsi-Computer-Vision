import cv2
import math
import numpy as np
import os
from skimage import feature

HUE_THRESH = 0
SAT_THRESH = 0
CANNY_MIN = 0
CANNY_MAX = 255
SOBEL_SIZE = 5

HUE_MIN = 0
HUE_MAX = 1

LBP_POINTS = 1
LBP_RADIUS = 1

pred_dir = "tools/dataset/predict/"
namefile = "result2021-07-08-135845g1b2.jpg"

win_name = "Weld Project"
# Read image in BGR colorspace
image = cv2.imread(os.path.join(pred_dir,namefile),cv2.IMREAD_COLOR)
# image = cv2.imread('test.jpg',cv2.IMREAD_COLOR)
# image = cv2.imread('travel_arc_ampere_weld.jpg',cv2.IMREAD_COLOR)
# image = cv2.imread('bad_weld.jpg',cv2.IMREAD_COLOR)

# Resize image
dim = (400,300)
image = cv2.resize(image,dim,interpolation=cv2.INTER_AREA)

# Convert to HSV
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
print(gray.shape)
gray = np.expand_dims(gray,axis=2)
print(gray.shape)

# Save each element of HSV
h,s,v = cv2.split(image)

# Threshold using Otsu method
blur_hue = cv2.GaussianBlur(h,(5,5),0)
blur_sat = cv2.GaussianBlur(s,(5,5),0)
otsu_hue,th_hue = cv2.threshold(blur_hue,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
otsu_sat,th_sat = cv2.threshold(blur_sat,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
HUE_THRESH = otsu_hue
SAT_THRESH = otsu_sat
print("OTSU VALUE = ",HUE_THRESH,SAT_THRESH)

# Create trackbar
def nothing(val):
    pass

# cv2.namedWindow(win_name)
# cv2.createTrackbar("HUE = ",win_name,0,255,nothing)
# cv2.createTrackbar("SAT = ",win_name,0,255,nothing)
# cv2.createTrackbar("CANNY_MIN = ",win_name,0,255,nothing)
# cv2.createTrackbar("CANNY_MAX = ",win_name,0,255,nothing)
# cv2.createTrackbar("SOBEL_SIZE = ",win_name,1,31,nothing)
# cv2.createTrackbar("HUE_MIN = ",win_name,0,180,nothing)
# cv2.createTrackbar("HUE_MAX = ",win_name,0,180,nothing)
# cv2.createTrackbar("LBP_POINTS = ",win_name,1,100,nothing)
# cv2.createTrackbar("LBP_RADIUS = ",win_name,1,100,nothing)


def get_pixel(img, center, x, y):
    new_value = 0  
    try:
        if img[x][y] >= center:
            new_value = 1            
    except:
        pass    
    return new_value

# Function for calculating LBP
def lbp_calculated_pixel(img, x, y):  
    center = img[x][y]
    
    bit_val = get_pixel(img, center, x-1, y-1)
    bit_val = bit_val << 1 | get_pixel(img, center, x-1, y)
    bit_val = bit_val << 1 | get_pixel(img, center, x-1, y + 1)
    bit_val = bit_val << 1 | get_pixel(img, center, x, y + 1)
    bit_val = bit_val << 1 | get_pixel(img, center, x + 1, y + 1)
    bit_val = bit_val << 1 | get_pixel(img, center, x + 1, y)
    bit_val = bit_val << 1 | get_pixel(img, center, x + 1, y-1)
    bit_val = bit_val << 1 | get_pixel(img, center, x, y-1)

    return bit_val

img_lbp = np.zeros((gray.shape[0], gray.shape[1]),
                   np.uint8)

for i in range(0, gray.shape[0]):
    for j in range(0, gray.shape[1]):
        img_lbp[i, j] = lbp_calculated_pixel(gray, i, j)


# # Show
# while(True):
#     HUE_THRESH = cv2.getTrackbarPos("HUE = ",win_name)
#     SAT_THRESH = cv2.getTrackbarPos("SAT = ",win_name)
#     CANNY_MIN = cv2.getTrackbarPos("CANNY_MIN = ",win_name)
#     CANNY_MAX = cv2.getTrackbarPos("CANNY_MAX = ",win_name)
#     SOBEL_SIZE = cv2.getTrackbarPos("SOBEL_SIZE = ",win_name)
#     HUE_MAX = cv2.getTrackbarPos("HUE_MAX = ",win_name)
#     HUE_MIN = cv2.getTrackbarPos("HUE_MIN = ",win_name)
#     LBP_POINTS = cv2.getTrackbarPos("LBP_POINTS = ",win_name)
#     LBP_RADIUS = cv2.getTrackbarPos("LBP_RADIUS = ",win_name)
    

#     # Code here
#     # Threshold using Binary method
#     binary_hue,result_hue = cv2.threshold(blur_hue,HUE_THRESH,255,cv2.THRESH_BINARY)
#     binary_sat,result_sat = cv2.threshold(blur_sat,SAT_THRESH,255,cv2.THRESH_BINARY)

#     # Canny Edge
#     edges = cv2.Canny(blur_hue,CANNY_MIN,CANNY_MAX)

#     # Sobel Y
#     if SOBEL_SIZE % 2 == 0:
#         SOBEL_SIZE -= 1
#     sobel_hue = cv2.Sobel(blur_hue,cv2.CV_64F,0,1,ksize=SOBEL_SIZE)
#     sobel_sat = cv2.Sobel(blur_sat,cv2.CV_64F,0,1,ksize=SOBEL_SIZE)

#     # InRange method
#     lower_color = np.array([HUE_MIN,0,0])
#     upper_color = np.array([HUE_MAX,255,255])
#     mask = cv2.inRange(cv2.cvtColor(image,cv2.COLOR_BGR2HSV),lower_color,upper_color)

#     # LBP Image
#     lbp = feature.local_binary_pattern(gray,LBP_POINTS,LBP_RADIUS, method="ror")

#     # Merge
#     image = cv2.merge([h,s,v])

#     # Show
#     cv2.imshow("Original",cv2.cvtColor(image,cv2.COLOR_HSV2BGR))
#     cv2.imshow("Edge",edges)
#     cv2.imshow('Mask InRange', mask)
#     cv2.imshow("lbp custom",img_lbp)
#     # cv2.imshow(win_name+' Otsu',th)
#     cv2.imshow(win_name+' HUE',result_hue)
#     # cv2.imshow(win_name+' SAT',result_sat)
#     # cv2.imshow(win_name+' SOBH',sobel_hue)
#     # cv2.imshow(win_name+' SOBS',sobel_sat)
#     cv2.imshow("LBP",lbp)


#     # Log
#     if(HUE_THRESH):
#         # print(SOBEL_SIZE)
#         # print(math.floor((SOBEL_SIZE+1)/2+1))
#         pass



#     if cv2.waitKey(100) == ord("q") or 0xff == 27:
#         print("=============")
#         print(f"Threshold Binary Hue = {binary_hue}")
#         print(f"Threshold Binary Sat = {binary_sat}")
#         print(f"InRange Min = {HUE_MIN}")
#         print(f"InRange Max = {HUE_MAX}")
#         break

# cv2.destroyAllWindows()
