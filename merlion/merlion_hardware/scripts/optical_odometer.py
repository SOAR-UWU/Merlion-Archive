#!/usr/bin/env python2
import rospy
import cv2
import numpy as np
import time
from std_msgs.msg import Float64
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge


class OpticalOdometer:
    def __init__(self):
        rospy.init_node('optical_odometer')

        self.camera_hfov = float(rospy.get_param('~camera_hfov'))
        self.pool_depth = float(rospy.get_param('~pool_depth'))
        self.window_width = int(rospy.get_param('~window_width'))
        self.window_height = None
        self.odom_frame = rospy.get_param('~odom_frame')
        self.base_link_frame = rospy.get_param('~base_link_frame')
        self.debug = bool(rospy.get_param('~debug'))

        self.bridge = CvBridge()
        rospy.Subscriber('image', Image, self.on_new_frame)
        self.depth = 0
        rospy.Subscriber('depth', Float64, self.on_new_depth)
        self.odom_pub = rospy.Publisher('odometer', Odometry, queue_size=10)

    def on_new_depth(self, msg):
        self.depth = msg.data

    def on_new_frame(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        frame = np.array(frame, dtype=np.uint8)
        if self.window_height is None:
            self.window_height = frame.shape[0] * self.window_width // frame.shape[1]
        self.optical_flow(cv2.resize(frame, (self.window_width, self.window_height)))

    def optical_flow(self, frame):
        # On first frame
        if not hasattr(self, 'prev_gray'):
            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.prev_time = time.time()
            return

        # Calculates dense optical flow by Farneback method
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        self.prev_gray = gray

        # Updates time
        period = time.time() - self.prev_time
        self.prev_time = time.time()

        # Calculates average speed
        avg_vel = np.average(np.average(flow, 0), 0)
        vy, vx = avg_vel / period * \
            2 * (self.pool_depth - self.depth) * np.tan(self.camera_hfov / 2) / self.window_width

        # Publish states
        odom_msg = Odometry()
        odom_msg.header.stamp = rospy.Time.now()
        odom_msg.header.frame_id = self.odom_frame
        odom_msg.child_frame_id = self.base_link_frame
        odom_msg.twist.twist.linear.x = vx
        odom_msg.twist.twist.linear.y = vy
        odom_msg.pose.pose.position.z = -self.depth

        self.odom_pub.publish(odom_msg)

        if self.debug:
            # Computes the magnitude and angle of the 2D vectors
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            # Sets image hue according to the optical flow direction
            mask = np.zeros_like(frame)
            mask[..., 1] = 255
            mask[..., 0] = angle * 180 / np.pi / 2
            # Sets image value according to the optical flow magnitude (normalized)
            mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            # Converts HSV to RGB (BGR) color representation
            rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
            cv2.imshow('input', frame)
            cv2.imshow('dense optical flow', rgb)
            cv2.waitKey(1)
            print('vx:{:4.1f}\tvy:{:4.1f}\t'.format(vx, vy))


if __name__ == '__main__':
    OpticalOdometer()
    rospy.spin()
