#!/usr/bin/env python


import time
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

t = time.time()
taken_picture = False

def take_picture(image):
    current_time = time.time()
    global taken_picture, t
    if current_time - t > 10  and taken_picture == False:
        bridge = CvBridge()
        frame = bridge.imgmsg_to_cv2(image, "bgr8")
        frame = np.array(frame, dtype=np.uint8)
	str_time = str(current_time)
        cv2.imwrite("/home/soar/Pictures/img_test22.png", frame)
        t = current_time
        taken_picture = True
        print(frame)
        print("Taken")

def snapshotter():
    rospy.init_node("snapshotter", anonymous=True)
    rospy.Subscriber("/merlion_hardware/front_camera/image_raw", Image, take_picture)



if __name__ == '__main__':
    snapshotter()
    rospy.spin()
