import cv2

class ColourFunctions:
    """Contains functions for processing images based on colour."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def _hue_distance(colour1, colour2):
        """Given two colours, calculate the difference in their hues. Accounts for 
        looping of the hue scale, i.e. hue given by 179 is similar to hue given by 0.
        
        ACCEPTS COLOURS IN HSV FORMAT"""
        return min(abs(colour1[0] - colour2[0]), abs(colour1[0] - colour2[0] - 180))

    @staticmethod
    def find_colour(frame, colour):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        if colour == "green" or colour == "g":
            green_range = (60,95)
            return cv2.inRange(hsv, (green_range[0],0,0), (green_range[1],255,255))

        elif colour == "yellow" or colour == "y":
            yellow_range = (55,90)
            return cv2.inRange(hsv, (yellow_range[0],0,0), (yellow_range[1],255,255))
        