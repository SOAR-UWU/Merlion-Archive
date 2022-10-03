"""Code to detect gates. Can detect both the qualification and red and green gate."""

# Red most differentiable from surroundings by value, as the colour does not travel well through water.
# In particularly strong sunlight (near the surface) red colour is observable
# but unlikely to be consistent.

# In this code, the blue channel is used instead of the value, as most of the light is blue anyway.

# Green most differentiable by colour, roughly same value as the surroundings.

# In future can implement some modelling of how the gate should look like based on the
# yaw of the camera, so it is detectable from any angle.

import math
from typing import Sequence
import cv2
import numpy as np
from geometry import Line
# import numpy.typing as npt
from math import sqrt
import itertools
import rospy
import cv_bridge as CvBridge
from vision_msgs import BoundingBox2D
from sensor_msgs import Image

class GateTracker:
    def __init__(self, ns = "/gate"):
        rospy.init_node('gate_tracker')

        # Create the cv_bridge object
        self.bridge = CvBridge()

        self.result_pub = rospy.Publisher('pos', BoundingBox2D, queue_size=10)
        self.image_sub = rospy.Subscriber('image', Image, self.on_new_frame)
        
        self.ORANGE = rospy.get_param(f"{ns}/orange")
        self.GREEN = rospy.get_param(f"{ns}/green")
        self.VERTICAL_LINE_THRESH = rospy.get_param(f"{ns}/vertical_line_thresh")
        self.HORIZONTAL_LINE_THRESH = rospy.get_param(f"{ns}/horizontal_line_thresh")
        self.CLUSTER_MAX_HEIGHT_DIFF = rospy.get_param(f"{ns}/line_cluster_max_height_diff")
        self.HOUGH_MIN_LINE_LEN = rospy.get_param(f"{ns}/hough_minLineLength")
        self.HOUGH_MAX_LINE_GAP = rospy.get_param(f"{ns}/hough_maxLineGap")
        self.CANNY1 = rospy.get_param(f"{ns}/canny_minVal")
        self.CANNY2 = rospy.get_param(f"{ns}/canny_maxVal")
        self.CANNY_APERTURE = rospy.get_param(f"{ns}/canny_apertureSize")
        self.GAUSSIAN = rospy.get_param(f"{ns}/gaussian_blur")
        self.DILATE = rospy.get_param(f"{ns}/dilate")
        self.VERTICAL_LEN_RATIO = rospy.get_param(f"{ns}/vertical_len_ratio")
        self.VERTICAL_MAX_HEIGHT_DIFF = rospy.get_param(f"{ns}/vertical_max_height_diff")
        self.VERTICAL_LINE_ANGLE = rospy.get_param(f"{ns}/vertical_line_angle")
        self.CORNER_MAX_GAP_PROPORTION = rospy.get_param(f"{ns}/corner_max_gap_proportion")
        self.DISTANCE_MAX_ERROR_PROPORTION = rospy.get_param(f"{ns}/distance_max_error_proportion")

    def on_new_frame(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        frame = np.array(frame, dtype=np.uint8)
        colour_lines = self.find_lines_from_colour(frame, self.ORANGE, self.GAUSSIAN, self.CANNY1, self.CANNY2, self.CANNY_APERTURE)
        gates = self.find_best_gates(colour_lines, self.QUALIFICATION_GATE)
        (x,y), (w,h), r = gates[0]
        
        box = BoundingBox2D()
        box.center.x = x
        box.center.y = y
        box.size_x = w
        box.size_y = h

        self.result_pub.publish(box)

    def find_lines_from_value(self, frame):

        b, _, _ = cv2.split(frame)        # just use blue channel for now, CHANGE IF NOT ENOUGH CONTRAST
        blurred = cv2.GaussianBlur(b, self.GAUSSIAN, 0)
        edges = cv2.Canny(blurred, self.CANNY1, self.CANNY2, apertureSize=self.CANNY_APERTURE)
        edges = cv2.dilate(edges,np.ones(self.DILATE, np.uint8))
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/360, 20, None,
                                self.HOUGH_MIN_LINE_LEN, self.HOUGH_MAX_LINE_GAP)
        lines = np.array(lines)
        lines = np.squeeze(lines)
        return lines

    def find_lines_from_colour(self, frame, colour_range):
        """Finds lines based on hue differences"""

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, _, _ = cv2.split(hsv)
        blurred = cv2.GaussianBlur(h, self.GAUSSIAN, 0)
        edges = cv2.Canny(blurred, self.CANNY1, self.CANNY2, apertureSize=self.CANNY_APERTURE)
        colour_mask = cv2.inRange(hsv, (colour_range[0],0,0), (colour_range[1],255,255)) 
        edges_masked = cv2.bitwise_and(edges,colour_mask)
        dilated = cv2.dilate(edges_masked, np.ones(self.DILATE, np.uint8))

        lines = cv2.HoughLinesP(dilated, 1, np.pi/360, 20, None,
                                self.HOUGH_MIN_LINE_LEN, self.HOUGH_MAX_LINE_GAP)

        return lines

    def check_vertical_line(self, line, bot_roll = np.array([[1, 0], [0, 1]])):
        """Check that a given line is vertical and long, accounting for rotation of bot
        
        line: 4-tuple with format (x1, y1, x2, y2)
        bot_roll: rotation matrix derived from roll of bot"""
        x1, y1, x2, y2 = line.cv_line()
        # might need to transpose
        x1_corrected, y1_corrected = bot_roll @ np.array((x1,y1))
        x2_corrected, y2_corrected = bot_roll @ np.array((x2,y2))

        line_corrected = Line(cv_line=[x1_corrected,y1_corrected,x2_corrected,y2_corrected])
        if abs(line_corrected.theta() - math.pi) < self.VERTICAL_LINE_ANGLE or abs(line_corrected.theta()) < self.VERTICAL_LINE_ANGLE:
            return True
        else: return False
    
    def check_horizontal_line(self, line, bot_roll = np.array([[1, 0], [0, 1]])):
        """Check that a given line is vertical and long, aaccounting for rotation of bot
        
        line: 4-tuple with format (x1, y1, x2, y2)
        bot_roll: rotation matrix derived from roll of bot"""
        x1, y1, x2, y2 = line.cv_line()
        # might need to transpose
        x1_corrected, y1_corrected = bot_roll @ np.array((x1,y1))
        x2_corrected, y2_corrected = bot_roll @ np.array((x2,y2))

        line_corrected = Line(cv_line=[x1_corrected,y1_corrected,x2_corrected,y2_corrected])
        if abs(line_corrected.theta() - math.pi / 2) < self.HORIZONTAL_LINE_THRESH or abs(line_corrected.theta() + math.pi / 2) < self.HORIZONTAL_LINE_THRESH:
            return True
        else: return False
        
    def _cluster_lines(lines):
        lines = list(lines)
        grouped = []
        while lines:
            main = lines[0]
            filter = []
            for line in lines:
                filter.append(Line.mergeable(line, main))
            grouped.append(list(itertools.compress(lines,filter)))
            lines = list(itertools.compress(lines, [not i for i in filter]))
        return grouped

    def group_by_length(self, lines: Sequence[Line]):
        grouped = []
        lines = list(lines)
        lines.sort(key = Line.length, reverse = True)
        while lines:
            main_length = lines[0].length()
            group = list(itertools.takewhile(
                lambda line: abs(line.length() - main_length) <= self.CLUSTER_MAX_HEIGHT_DIFF, lines))
            grouped.append(group)
            lines = lines[len(group):]  # remove the entries from the main list      
        return grouped

    def _find_vertical_gap(lines):
        def func(horz):
            _, _, _, top1 = lines[0].cv_line()
            _, _, _, top2 = lines[1].cv_line()
            _, horz_height = horz.centre()
            return abs((top1 + top2) / 2 - horz_height)
        return func  

    def _valid_horizontal(self, line1: Line, line2: Line, max_gap_proportion):
        max_gap = (line1.length() + line2.length()) / 2 * max_gap_proportion
        def func(horz_line):
            _, top1 = line1.points()
            _, top2 = line2.points()
            h_start, h_end = horz_line.points()
            if Line.dist(top1, h_start) <= max_gap \
                and Line.dist(top2, h_end) <= max_gap:
                return True
            elif Line.dist(top2, h_start) <= max_gap \
                and Line.dist(top1, h_end) <= max_gap:
                return True
            else: return False
        return func

    QUALIFICATION_GATE = 0
    GREEN_RED_GATE = 1

    # TODO change the max height diff to a proportional number
    def find_best_gates(self, lines, mode):
        lines = list(lines) # Convert ndarray to list to use list methods for sorting.
        lines = list(map(Line, lines))

        # First, look for the vertical (side) lines.
        verts = list(filter(GateTracker.check_vertical_line, lines))

        # Next, sort by some conditions.
        # These lines should be roughly the same length.
        grouped = GateTracker.group_by_length(verts)

        # First condition: There should be a pair of lines of the same length.

        grouped = list(filter(lambda group: len(group) >= 2, grouped))

        # Second condition: lines should be roughly at the same vertical position

        paired = []
        for group in grouped:
            for pair in itertools.combinations(group, 2):
                y1 = pair[0].centre()[1]
                y2 = pair[1].centre()[1]
                if abs(y1 - y2) <= self.VERTICAL_MAX_HEIGHT_DIFF:
                    paired.append(pair)

        # For the green and red gate, lines should be connected by a horizontal line at the top.

        if mode == GateTracker.GREEN_RED_GATE:
            horzs = list(filter(GateTracker.check_horizontal_line, lines))

            possible_gates = []

            for pair in paired:
                
                valid_horzs = list(filter(GateTracker._valid_horizontal(pair[0],pair[1],self.CORNER_MAX_GAP_PROPORTION), horzs))
                if valid_horzs:     # if valid horizontal lines are found, append to the pair of lines as a list.
                    valid_horzs.sort(key = GateTracker._find_vertical_gap(pair))
                    gate_lines = []
                    gate_lines.extend(pair)
                    gate_lines.append(valid_horzs[0])
                    possible_gates.append(gate_lines)
        
        # For the qualification gate, there is no horizontal bar connecting the two poles.

        elif mode == GateTracker.QUALIFICATION_GATE:
            possible_gates = list(filter(lambda pair: GateTracker._rate_distance(pair[0], pair[1]) <= self.DISTANCE_MAX_ERROR_PROPORTION, paired))
        
        # return possible_gates
        return [GateTracker._box_from_gate(gate) for gate in possible_gates]

    # QUALIFICATION_GATE = 1.5
    # GREEN_RED_GATE = 1.0
        
    def _rate_distance(self, line1: Line, line2: Line):
        """For a pair of lines, rate how much the distance between them tallies with
        the given ratio of horizontal to vertical distance.
        0 is a perfect score, higher score is worse.
        
        PARAMETERS
        
        line1, line2 - Line objects
        ratio - ratio of the horizontal distance to the height"""

        # Future improvement: Add in yaw value from imu to account for perspective

        ave_height = (line1.length() + line2.length()) / 2
        ave_distance = Line.mean_distance(line1, line2)
        return abs(ave_height * self.VERTICAL_LEN_RATIO - ave_distance) / ave_height

    @staticmethod
    def _box_from_gate(gate_lines):
        if len(gate_lines) == 2:    # in this case, the lines represent the two side verticals
            p1, p2 = gate_lines[0].points()
            p3, p4 = gate_lines[1].points()
        
        elif len(gate_lines) == 3:  # in this case, there are the two verticals and upper horizontal
            l1, l2, l3 = gate_lines
            p1 = Line.find_intersect(l1, l3)
            p2 = Line.find_intersect(l2, l3)
            p3, _ = l2.points()
            p4, _ = l1.points()
            pts = p1, p2, p3, p4
        
        return cv2.minAreaRect(pts)
