"""Code for detecting the position of the flares, both red and
yellow. Will return pixel positions of top and bottom."""
# RED and YELLOW flare - colours
# VERTICAL - compensate for craft rotation before checking
# STATIONARY

# TODO restructure to include continuous tracking

from math import sqrt
import math
from typing import Sequence
import cv2
import numpy as np
# import numpy.typing as npt
from geometry import Line
from processing_helpers import Direction
import rospy
import cv_bridge as CvBridge
from sensor_msgs.msg import Image
from vision_msgs import BoundingBox2D

class FlareDetector:
    def __init__(self, ns = "/flare"):
        rospy.init_node('gate_tracker')

        # Create the cv_bridge object
        self.bridge = CvBridge()

        self.result_pub = rospy.Publisher('flare_box', BoundingBox2D, queue_size=10)
        self.image_sub = rospy.Subscriber('image', Image, self.on_new_frame)

        self.YELLOW = rospy.get_param(f"{ns}/yellow")
        self.VERT_MAX_LEN_DIFF = rospy.get_param(f"{ns}/vert_max_len_diff")
        self.VERTICAL_LINE_THRESH = rospy.get_param(f"{ns}/vertical_line_thresh")
        self.HOUGH_MIN_LINE_LEN = rospy.get_param(f"{ns}/hough_minLineLength")
        self.HOUGH_MAX_LINE_GAP = rospy.get_param(f"{ns}/hough_maxLineGap")
        self.CANNY1 = rospy.get_param(f"{ns}/canny_minVal")
        self.CANNY2 = rospy.get_param(f"{ns}/canny_maxVal")
        self.CANNY_APERTURE = rospy.get_param(f"{ns}/canny_apertureSize")
        self.GAUSSIAN = rospy.get_param(f"{ns}/gaussian_blur")
        self.DILATE = rospy.get_param(f"{ns}/dilate")

    def on_new_frame(self, msg, mode):
        # put in mode checker to decide which flare to look for. Should be from ROS topic!

        ###################################
        # INCOMPLETE! Change this part later
        ###################################

        if mode:
            rang = self.GREEN

        ###################################
        # INCOMPLETE! Change this part later
        ###################################

        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        frame = np.array(frame, dtype=np.uint8)
        lines = self.find_lines_from_colour(frame, rang)
        verts = self.check_vertical_line(lines)
        (x,y), (w,h), r = self.find_best_flare(verts)

        box = BoundingBox2D()
        box.center.x = x
        box.center.y = y
        box.center.theta = r
        box.size_x = w
        box.size_y = h

        self.result_pub.publish(box)

    def find_lines_from_colour(self, frame, colour_range):
        """Finds lines based on hue differences"""

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, _, _ = cv2.split(hsv)
        blurred = cv2.GaussianBlur(h, self.GAUSSIAN)
        edges = cv2.Canny(blurred, self.CANNY1, self.CANNY2, apertureSize=self.CANNY_APERTURE)
        colour_mask = cv2.inRange(hsv, (colour_range[0],0,0), (colour_range[1],255,255)) 
        edges_masked = cv2.bitwise_and(edges,colour_mask)
        dilated = cv2.dilate(edges_masked, np.ones(self.DILATE, np.uint8))

        lines = cv2.HoughLinesP(dilated, 1, np.pi/360, 20, None,
                                self.HOUGH_MIN_LINE_LEN, self.HOUGH_MAX_LINE_GAP)

        return [Line(line) for line in lines]

    def check_vertical_line(self, line: Line, bot_roll = np.array([[1, 0], [0, 1]])):
        """Check that a given line is vertical and long, accounting for rotation of bot
        
        line: 4-tuple with format (x1, y1, x2, y2)
        bot_roll: rotation matrix derived from roll of bot"""
        x1, y1, x2, y2 = line.cv_line()
        # might need to transpose
        x1_corrected, y1_corrected = bot_roll @ np.array((x1,y1))
        x2_corrected, y2_corrected = bot_roll @ np.array((x2,y2))

        max_angle_error = self.VERTICAL_LINE_THRESH / 180 * math.pi
        line_corrected = Line(cv_line=[x1_corrected,y1_corrected,x2_corrected,y2_corrected])
        if abs(line_corrected.theta() - math.pi) < max_angle_error or abs(line_corrected.theta()) < max_angle_error:
            return True
        else: return False

    # class constants for setting output of flare search
    BOX = 0
    TOPNBOTTOM = 1
    CORNERS = 2

    @staticmethod
    # Retune max_length_diff!
    def find_best_flare(lines: Sequence[Line], max_length_diff = 10, mode = BOX):
        """Scores the possible lines marking the flare by length."""
        
        lines = list(lines) # Convert from ndarray to list to use the sort() method

        lines.sort(key = Line.length)

        if mode == FlareDetector.BOX:
            if abs(Line.length(lines[-1]) - Line.length(lines[-2])) <= max_length_diff:
                p1, p2 = lines[-1].points()
                p3, p4 = lines[-2].points()
                pts = np.array(p1, p2, p3, p4)
                (x,y), (w,h), r = cv2.minAreaRect(pts)
                
                # Correcting for OpenCV's weird rotation expression and converting to radians
                # https://theailearner.com/tag/cv2-minarearect/
                # Setting the smaller side as the first element in the dimensions tuple
                # Rotation measured clockwise. Vertical is 0.
                if w <= h:
                    r =  (90 + r) / 180 * math.pi
                    return (x,y), (w,h), r
                else:
                    r =  r / 180 * math.pi
                    return (x,y), (h,w), r


            else:
                x1, y1, x2, y2 = lines[-1].cv_line()
                centre = lines[-1].centre()
                h = Line.length(lines[-1])
                w = 1
                r = math.pi - lines[-1].theta
            
            return centre, (w,h), r

        elif mode == FlareDetector.TOPNBOTTOM:
            if abs(Line.length(lines[-1]) - Line.length(lines[-2])) <= max_length_diff:
                line1_x1, line1_y1, line1_x2, line1_y2 = Line.cv_line(lines[-1])
                line2_x1, line2_y1, line2_x2, line2_y2 = Line.cv_line(lines[-2])
                top = ((line1_x1 + line2_x1)/2, (line1_y1 + line2_y1)/2)
                bottom = ((line1_x2 + line2_x2)/2, (line1_y2 + line2_y2)/2)
            else:
                x1, y1, x2, y2 = Line.cv_line(lines[-1])
                top = (x1, y1)
                bottom = (x2, y2)

            return (Direction.get_direction(top), Direction.get_direction(bottom))
        
        elif mode == FlareDetector.CORNERS:
            if abs(Line.length(lines[-1]) - Line.length(lines[-2])) <= max_length_diff:
                x1, y1, x2, y2 = Line.cv_line(lines[-1])
                x4, y4, x3, y3 = Line.cv_line(lines[-2])
                c1, c2, c3, c4 = (x1,y1), (x2,y2), (x3,y3), (x4,y4)
                
                return (Direction.get_direction(c1), Direction.get_direction(c2), Direction.get_direction(c3), Direction.get_direction(c4))

            else:
                x1, y1, x2, y2 = Line.cv_line(lines[-1])
                c1, c2 = (x1,y1), (x2,y2)

                return (Direction.get_direction(c1), Direction.get_direction(c2))

        else:
            raise ValueError(f"Inapprpriate setting for scoring flare used in {__name__}!")
