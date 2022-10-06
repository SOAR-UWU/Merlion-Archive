# CV Scripts

This folder contains all the Python files for CV in the bot.

## Custom Libraries
`geometry/Line` - defines all functions for dealing with line information in CV. Can store in the format of OpenCV lines `(x1, y1, x2, y2)`, two points `((x1, y1), (x2, y2))`, or `centre / angle / length`.

`processing_helpers/ColourFunctions` - defines functions for working with colour values. Currently unused.

`processing_helpers/Direction` - defines functions for retrieving the direction vector of a given point with respect to the camera. Currently unused.

## Scripts

`bucket.py`, `flare.py`, and `gate.py` - scripts for detecting the corresponding obstacles.

`calibrate.py` - script to get camera intrinsics. Ripped from [this article](https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0).

`hue_tuning.py` - script to get hue values for colour calibration. Follow instructions in the script. This script should be run on your computer, not the bot!

`camera.py` - unfinished script for uploading videos to Google Drive. Done by seniors.
