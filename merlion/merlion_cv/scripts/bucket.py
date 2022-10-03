import itertools
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1st step: Find mat from far away

####This Code tests bounding box on the green mat
#-hex colour picker>convert to hsv value>upper lower bounds>gaussian blur>bounding box

class BucketDetector:
    def __init__(self, img,
    green_range = [67,90], 
    erode_extent = 4, 
    green_threshold = 0.2,
    min_bucket_area = 20000,   # Rmb to tune this number!  
    max_ball_area = 0.1,
    min_ball_area = 0.01,
    white_value = [200,255]
    ):
        self.img = img
        self.green_range = green_range
        self.erode_extent = erode_extent
        self.green_threshold = green_threshold
        self.min_bucket_area = min_bucket_area
        self.max_ball_area = max_ball_area
        self.min_ball_area = min_ball_area
        self.white_range = white_value

    @staticmethod
    def _create_green_mask(self):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (self.green_range[0],0,0), (self.green_range[1],255,255))
        e_kernel = np.ones((self.erode_extent,self.erode_extent), np.uint8)
        return cv2.erode(mask, e_kernel)
    
    def _create_white_mask(self):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0,180,self.white_range[0]),(0,180,self.white_range[1]))
        e_kernel = np.ones((self.erode_extent,self.erode_extent),np.uint8)
        return cv2.erode(mask)

    RECTANGLE = 0
    CONTOUR = 1
    CENTRE = 2
    OUT_MAT = 0
    OUT_CONTOURS = 1
    def find_green(self, output_object = OUT_MAT, output_mode = CENTRE, ctr_mode = cv2.RETR_TREE):
        """Returns a rotated bounding rectangle, of the form ((centre_x, centre_y), (width, height), rotation)"""
        
        green = self._create_green_mask(self)
        contours, hierarchy = cv2.findContours(green, ctr_mode, 1)

        if output_object == self.OUT_CONTOURS:
            return contours, hierarchy

        elif output_object == self.OUT_MAT:
            # First, find the area of all contours.
            cnt_areas = [(contour, cv2.contourArea(contour)) for contour in contours]
            cnt_areas.sort(key = lambda x: x[1])

            if output_mode == self.CONTOUR:
                return cnt_areas[-1][0]
            else:
                # After this sort, the contour with the largest area will be found in the last element.
                # Use OpenCV boundingRect function to get the details of the contour
                
                rect = cv2.minAreaRect(cnt_areas[-1][0])
                # rect has the form ((x,y), (w,h), r)

                if output_mode == self.RECTANGLE:
                    return rect

                elif output_mode == self.CENTRE:
                    return rect[0]
            
    def find_white(self, output_object = OUT_MAT, output_mode = CENTRE, ctr_mode = cv2.RETR_TREE):
        """Returns a rotated bounding rectangle, of the form ((centre_x, centre_y), (width, height), rotation)"""
        
        white = self._create_white_mask(self)
        contours, hierarchy = cv2.findContours(white, ctr_mode, 1)

        if output_object == self.OUT_CONTOURS:
            return contours, hierarchy

        elif output_object == self.OUT_MAT:
            # First, find the area of all contours.
            cnt_areas = [(contour, cv2.contourArea(contour)) for contour in contours]
            cnt_areas.sort(key = lambda x: x[1])

            if output_mode == self.CONTOUR:
                return cnt_areas[-1][0]
            else:
                # After this sort, the contour with the largest area will be found in the last element.
                # Use OpenCV boundingRect function to get the details of the contour
                
                rect = cv2.minAreaRect(cnt_areas[-1][0])
                # rect has the form ((x,y), (w,h), r)

                if output_mode == self.RECTANGLE:
                    return rect

                elif output_mode == self.CENTRE:
                    return rect[0]

    def find_white(self, output_object = OUT_MAT, output_mode = CENTRE, ctr_mode = cv2.RETR_TREE):
        """Returns a rotated bounding rectangle, of the form ((centre_x, centre_y), (width, height), rotation)"""
        
        white = self._create_white_mask(self)
        contours, hierarchy = cv2.findContours(white, ctr_mode, 1)

        if output_object == self.OUT_CONTOURS:
            return contours, hierarchy

        elif output_object == self.OUT_MAT:
            # First, find the area of all contours.
            cnt_areas = [(contour, cv2.contourArea(contour)) for contour in contours]
            cnt_areas.sort(key = lambda x: x[1])

            if output_mode == self.CONTOUR:
                return cnt_areas[-1][0]
            else:
                # After this sort, the contour with the largest area will be found in the last element.
                # Use OpenCV boundingRect function to get the details of the contour
                
                rect = cv2.minAreaRect(cnt_areas[-1][0])
                # rect has the form ((x,y), (w,h), r)

                if output_mode == self.RECTANGLE:
                    return rect

                elif output_mode == self.CENTRE:
                    return rect[0]

    def bottom_cam_find_green(self, img):
        green = self._create_green_mask()
        green_pxls = np.sum(green == 255)  # find number of green pixels detected
        total_pxls = np.size(img)
        if green_pxls / total_pxls > self.green_threshold:
            return True


    def correct_height_lines(img):    # version with finding both sides of mat. Use depth sensor instead?
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_threshold_green= (60, 0, 0) #100deg hue angle, 50% saturation and 50% value
        upper_threshold_green = (88, 255, 255) #!!!! Hue angle values between 80-90 depending how much the water "blueifys" the image, 100% saturation and 100% value
        green_mask = cv2.inRange(hsv_img, lower_threshold_green, upper_threshold_green) #this is the masked image

    def find_yaw_error(self):
        """Returns the angle that the BOT SHOULD TURN in order to align with the mat.
        This angle is positive in the anti-clockwise direction.
        To find the angle that the mat is offset by, just take the negative angle."""
        (_, (w,h), r) = self.find_green(output_mode = self.RECTANGLE)
        if w <= h:
            return (90 + r) / 180 * math.pi
        else:
            return r / 180 * math.pi

        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # colour_mask = cv2.inRange(hsv, (colour_range[0],0,0), (colour_range[1],255,255))
        # col_edges = cv2.Canny(colour_mask, 300, 500, apertureSize=5)
        # dilated = cv2.dilate(col_edges, np.ones((3, 3), np.uint8))

        # # Constants to put in params
        # minLineLength = 200
        # maxLineGap = 15

        # lines = cv2.HoughLinesP(dilated, 1, np.pi/360, 20, None,
        #                         minLineLength, maxLineGap)
        # return lines

    def correct_height_circles(frame, rad_diff_threshold):  # version with finding buckets
        circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, 1, 50)    # might need to change the default arguments if cannot
        if len(circles) < 2:
            return False
        radii = [circle[2] for circle in circles]
        radii.sort()
        pairs = list(itertools.pairwise(radii))
        diffs = [abs(pair[0] - pair[1]) for pair in pairs]
        diffs.sort()
        if diffs[0] <= rad_diff_threshold:
            return True
        else:
            return False

    def find_buckets(self):
        ctrs, hierarchy = self.find_green(output_object=BucketDetector.OUT_CONTOURS, output_mode = BucketDetector.CONTOUR, ctr_mode = cv2.RETR_TREE)
        hierarchy = hierarchy[0]
        hier_filter = [hier[3] > 0 for hier in hierarchy]
        nested_ctrs = list(itertools.compress(ctrs, hier_filter))
        ctrs_filter = [cv2.contourArea(ctr) > self.min_bucket_area for ctr in nested_ctrs]
        bucket_ctrs = list(itertools.compress(nested_ctrs,ctrs_filter))
        bucket_rects = [cv2.boundingRect(ctr) for ctr in bucket_ctrs]
        bucket_centres = [rect[0] for rect in bucket_rects]
        return bucket_centres, bucket_ctrs

    def find_balls(self):
        ctrs, hierarchy = self.find_green(output_object=BucketDetector.OUT_CONTOURS, output_mode = BucketDetector.CONTOUR, ctr_mode = cv2.RETR_TREE)
        hierarchy = hierarchy[0]
        hier_filter = [hier[3] > 0 for hier in hierarchy]
        nested_ctrs = list(itertools.compress(ctrs, hier_filter))
        balls_ctrs = []
        balls_centres = []
        for ball_ctr in nested_ctrs:
            ball_area = cv2.contourArea(ball_ctr)
            ball_rect = cv2.boundingRect(ball_ctr)
            ball_centre = ball_rect[0]
            if ball_area<self.max_ball_area and ball_area>self.min_ball_area and cv2.pointPolygonTest(ball_ctr,ball_centre,False) == 1:
                balls_ctrs.append(ball_ctr)
                balls_centres.append(ball_centre)
        return balls_centres, balls_ctrs
                
        


if __name__ == "__main__":
    img = cv2.imread("UWU_Navigations\Obstacle_detection\Bucket Detection\\top_view_cropped.jpg")
    scale = 0.5         # downscale image
    img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)), interpolation=cv2.INTER_AREA)
    bkt_detector = BucketDetector(img, min_bucket_area=500)

    bkts, ctrs = bkt_detector.find_buckets()
    print(bkts)
    cv2.drawContours(img, ctrs, -1, (255,0,0), 3)

    cv2.imshow("test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
