import cv2
import os

file_list = os.listdir('captured_image/')
file_list.sort()

imgs = []
for img_name in file_list:
    print(img_name)
    img = cv2.imread(os.path.join("captured_image", img_name))
    imgs.append(img)

stitcher = cv2.Stitcher.create(1)
status, pano = stitcher.stitch(imgs)

cv2.imwrite('result.jpg', pano)
