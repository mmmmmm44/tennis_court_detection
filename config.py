# Constants for "Robust Camera Calibration for Sport Videos using Court Models"

###################################
# Tennis Court
###################################

##############################
# 3.1 White Pixel Detection
##############################

# court line detection
TAU = 8
THRESHOLD_L = 128
THRESHOLD_D = 20

# textured regions
BLOCK_SIZE = 21
APERATURE_SIZE = 3

###################################
# 3.2 Court Line Candidate Detector
###################################

# [0, 1]
HOUGH_THRESHOLD = 0.42


# 3.2.1
THRESHOLD_R = 6

# check for duplicates
DEGREE_THRESH = 1
DIST_LINES_THRESH = 1.5

###################################
# 3.3.1 Finding Line Correspondences
###################################

CLS_ANGLE_THRESH = 25