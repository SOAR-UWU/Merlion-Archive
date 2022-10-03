import cv2
import numpy as np

sources = ['UWU_Navigations\Obstacle_detection\Bucket Detection\\top_view_cropped.jpg']

source = sources[0]
img = cv2.imread(source)

scale = 0.5         # downscale image
img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)), interpolation=cv2.INTER_AREA)

hue_range = [0,0]
erode_extent = 1

cv2.imshow("original", img)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

Q = 113
W = 119
LEFT_BRACKET = 91
RIGHT_BRACKET = 93
LEFT_ANGLE_BRACKET = 44
RIGHT_ANGLE_BRACKET = 46
ESC = 27

while True:
    # test = h.copy()
    mask = cv2.inRange(hsv, (hue_range[0],0,0), (hue_range[1],255,255))
    e_kernel = np.ones((erode_extent,erode_extent), np.uint8)
    out = cv2.erode(mask, e_kernel)
    cv2.imshow(source, out)
    print(hue_range)
    print(erode_extent)
    k = cv2.waitKey(0)
    if k == Q and hue_range[0] > 0:
        hue_range[0] -= 1
    elif k == W and hue_range[0] < hue_range[1]:
        hue_range[0] += 1
    elif k == LEFT_BRACKET and hue_range[1] > hue_range[0]:
        hue_range[1] -= 1
    elif k == RIGHT_BRACKET and hue_range[1] < 179:
        hue_range[1] += 1
    elif k == LEFT_ANGLE_BRACKET and erode_extent > 1:
        erode_extent -= 1
    elif k == RIGHT_ANGLE_BRACKET:
        erode_extent += 1
    elif k == ESC:
        break