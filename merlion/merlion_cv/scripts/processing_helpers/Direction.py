"""Get the bearing of a point from the perspective of the camera (direction vector)"""
# Adjust the parameters based on the intrinsics of the camera

import numpy as np

# These values are obtained from OpenCV camera calibration code.
# Replace this bit with ros params later
focal_x = 486.72551119965846
focal_y = 487.5417691189848
offset_x = 389.4484210209744
offset_y = 307.3813731845398
# Replace this bit with ros params later

class Direction:
    def get_direction(position):
        """Returns the direction vector of the pixel in terms of the camera coordinates."""
        # For these calculations, refer to the projection equation for cameras.
        # Assumming Z = 1 for these calculations, just to find the unit vector.
        x = (position[0] - offset_x) / focal_x
        y = (position[1] - offset_y) / focal_y

        direction = np.array([x,y,1])
        return direction / np.linalg.norm(direction)
