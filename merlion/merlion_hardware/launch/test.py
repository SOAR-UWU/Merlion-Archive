#! /usr/bin/env python2
# -*- coding: utf-8 -*-

import cv2
import sys
if __name__ == "__main__":
    resource = sys.argv[1]
    if len(resource) < 3: 
        resource_name = '/dev/video'
        resource = int(resource) 
    else: 
        resource_name = resource 
    print("Trying to open resource, ", resource_name) 
    cap = cv2.VideoCapture(resource)
    if not cap.isOpened():
        exit(0)
    rval, frame = cap.read() 
    while rval:
        cv2.imshow("Stream: " + resource_name, frame) 
        rval, frame = cap.read() 
        key = cv2.waitKey(20) 
        if key == 27 or key == 1048603:
            break 
    cv2.destroyWindow('preview')
