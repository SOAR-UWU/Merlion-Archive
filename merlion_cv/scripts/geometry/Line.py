import math
from typing_extensions import Self
import numpy as np
import itertools



# Might convert this to nonstatic class later

class Line:
    def __init__(self, cv_line = None, points = None, centre = None, angle = None, length = None):
        if cv_line is not None:
            self.x1,self.y1,self.x2,self.y2 = cv_line
        elif points:
            pt1, pt2 = points
            self.x1,self.y1 = pt1
            self.x2,self.y2 = pt2
        elif centre and angle is not None and length:
            from_centre = np.array([length/2*math.sin(angle), length/2*math.cos(angle)])
            pt1 = np.array(centre) + from_centre
            pt2 = np.array(centre) + from_centre
            self.x1,self.y1 = pt1
            self.x2,self.y2 = pt2
        self._reorient_line()

    def __getitem__(self, index):
        return self.cv_line()[index]

    def cv_line(self):
        return np.array([self.x1,self.y1,self.x2,self.y2])

    def points(self):
        pt1 = np.array([self.x1,self.y1])
        pt2 = np.array([self.x2,self.y2])
        return np.array([pt1,pt2])

    def _reorient_line(self):
        """Make sure lines are drawn from bottom up"""
        _, y1, _, y2 = self.cv_line()
        if y1 < y2:
            self.reverse()

    def reverse(self):
        self.x2,self.y2,self.x1,self.y1 = self.x1,self.y1,self.x2,self.y2

    @staticmethod
    def merge(lines):
        """Construct a new line from the extreme points of two or more lines."""
        x_coords = []
        y_coords = []
        for line in lines:
            x1, y1, x2, y2 = line.cv_line()
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
        x1, y1, x2, y2 = lines[0].cv_line()
        if x1 < x2:
            pt1 = (min(x_coords),max(y_coords))
            pt2 = (max(x_coords),min(y_coords))
        else:
            pt1 = (max(x_coords),max(y_coords))
            pt2 = (min(x_coords),min(y_coords))
        
        return Line(points = [pt1,pt2])

    @staticmethod
    def sign(x) -> int:
        if x < 0:
            return -1
        elif x > 0:
            return 1
        else:
            return 0

    @staticmethod
    def closest_distance(line1, line2):
        """Returns the closest distance between 2 lines."""
        n1, D1 = Line.line_equation(line1)
        n2, D2 = Line.line_equation(line2)

        x11,y11,x21,y21 = line1
        x12,y12,x22,y22 = line2

        l1p1 = np.array([x11,y11])
        l1p2 = np.array([x21,y21])
        l2p1 = np.array([x12,y12])
        l2p2 = np.array([x22,y22])

        # Find the dot products of the corners of the lines with the other line.

        dp1 = l1p1.dot(n2) - D2
        dp2 = l1p2.dot(n2) - D2
        dp3 = l2p1.dot(n1) - D1
        dp4 = l2p2.dot(n1) - D1

        # If the two lines intersect, the dot products of the two corners on each line
        # should be of opposite sign from each other.
        # In this case, the shortest distance is 0.

        if Line.sign(dp1) == - Line.sign(dp2) and Line.sign(dp3) == - Line.sign(dp4):
            # print(0)
            return 0.0

        # If the two lines do not intersect, the closest distance is either the
        # smallest dot product or the closest distance between two points.

        # If the perpendicular line of projection from any of the points lies on the other
        # line, then the closest distance is the smallest dot product from within
        # those points.

        # First, find the direction vectors of the lines.
        dir1 = l1p2 - l1p1
        dir2 = l2p2 - l2p1

        # Next, find the limits along these directions which defines each line.
        lim1 = (l1p1.dot(dir1), l1p2.dot(dir1))
        lim2 = (l2p1.dot(dir2), l2p2.dot(dir2))

        # If a point on the other line dotted with the direction vector gives a
        # value within the limits, its point of projection lies on that line.

        prod1 = l1p1.dot(dir2)
        prod2 = l1p2.dot(dir2)
        prod3 = l2p1.dot(dir1)
        prod4 = l2p2.dot(dir1)

        within_range = (
            Line._within(lim2,prod1),
            Line._within(lim2,prod2),
            Line._within(lim1,prod3),
            Line._within(lim1,prod4))
        
        dps = [dp1, dp2, dp3, dp4]
        valid_dps = list(itertools.compress(dps,within_range))
        if valid_dps:   # if not empty
            valid_dps.sort()
            # print("dot prod", abs(valid_dps[0]))
            return abs(valid_dps[0])
        
        # if there are none found, the closest distance is the shortest 
        # distance between two points.

        distances = [
            Line.dist(l1p1,l2p1),
            Line.dist(l1p1,l2p2),
            Line.dist(l1p2,l2p1),
            Line.dist(l1p2,l2p2)]
        distances.sort()
        # print("distance", abs(distances[0]))
        return abs(distances[0])
        
    @staticmethod
    def dist(point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    @staticmethod
    def _within(range, value):
        if value >= range[0] and value <= range[1] or value >= range[1] and value <= range[0]:
            return True
        else:
            return False

    @staticmethod
    def mergeable(line1, line2, max_angle_diff = 3/180*math.pi, max_position_difference = 10):
        if Line.closest_distance(line1,line2) <= max_position_difference \
            and abs(Line.angle_between(line1,line2)) <= max_angle_diff:
            return True
        else: return False
    
    def line_equation(self):
        """Returns the unit normal vector and the distance from origin.
        (n and D from r . n = D)"""
        pt1, pt2 = self.points()
        direction_vector = np.array(pt2 - pt1)

        # For a perpendicular vector, n1 / n2 = - d2 / d1
        if direction_vector[0] == 0:
            n = np.array([1,0], np.float64)
        else:
            n = np.array([-direction_vector[1]/direction_vector[0], 1])

        # Set to unit vector:
        n /= np.linalg.norm(n)

        # Find distance D:
        D = pt1.dot(n)

        return (n, D)

    def length(self) -> float:
        x1, y1, x2, y2 = self.cv_line()
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def theta(self) -> float:
        """Return angle of line with vertical. Note the angle is
        measured anticlockwise from the downward vertical as OpenCV 
        has reversed y-axis from usual coordinates."""
        x1, y1, x2, y2 = self.cv_line()
        if y2 - y1 == 0:
            return math.pi / 2
        else:
            return math.atan((x2 - x1)/(y2 - y1))

    def centre(self):
        x1, y1, x2, y2 = self.cv_line()
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    @staticmethod
    def mean_distance(line1: Self, line2: Self) -> float:
        dist_vec = line1.centre() - line2.centre()
        return math.sqrt(dist_vec[0] ** 2 + dist_vec[1] ** 2)

    @staticmethod
    def angle_between(line1: Self, line2: Self) -> float:
        return abs(line1.theta() - line2.theta())

    @staticmethod
    def find_intersect(line1: Self, line2: Self):
        """Finds the intersection between two lines.
        WARNING: This does not check if the point of intersection
        actually lies on the lines."""

        n1, D1 = line1.line_equation()
        n2, D2 = line2.line_equation()
        a, b = n1
        c ,d = n2
        x = (D1 - D2 * a * b + D1 * c * b) / (a * (d * a - b * c))
        y = (D2 * a - D1 * c) / (d * a - b * c)
        return (x, y)

if __name__ == "__main__":
    pass
    