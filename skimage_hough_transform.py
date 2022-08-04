# Ref: https://scikit-image.org/docs/dev/auto_examples/edges/plot_line_hough_transform.html

import cv2
import numpy as np

from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage import img_as_ubyte, img_as_float
from skimage.draw import line
from skimage import data

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.image as img

from line import Line

from PIL import Image

# try with tennis court image
def tennis_court():
    # tennis court image
    img = cv2.imread('tennis_pic_06.png')
    height, width, _ = img.shape
    if height > 960:
        w_h_ratio = width / height
        img = cv2.resize(img, (int(960 * w_h_ratio), 960), interpolation=cv2.INTER_AREA)
        height, width, _ = img.shape

    ###################################
    # 3.1 White Pixel Detection
    ###################################
    img_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    img_y = img_ycbcr[:, :, 0]

    # prevent overflow error
    img_y_int32 = img_ycbcr[:, :, 0].astype(np.int32)
    court_line_candidate = np.zeros((height, width))

    # threshold_l affects the most
    tau = 8
    threshold_l = 128
    threshold_d = 20

    for x in range(tau, width - tau):
        for y in range(tau, height - tau):
            court_line_candidate[y, x] = court_line_formula(
                    img_y_int32, y, x, tau, threshold_l, threshold_d
                )


    # exclude pixels that are in textured regions
    
    # previously implemented CourtLinePixelDetector::computeStructureTensorElements from https://github.com/gchlebus/tennis-court-detection
    # result: not as good as mine

    block_size = 17                 # affect the most in edge/ surface detection
    aperture_size = 3
    structure_matrix = cv2.cornerEigenValsAndVecs(
        img_y,
        block_size,
        aperture_size
    )

    line_structure_const = np.zeros((height, width))
    for x in range(width):
        for y in range(height):
            lambda_max, lambda_min = max(structure_matrix[y, x, 0], structure_matrix[y, x, 1]), min(structure_matrix[y, x, 0], structure_matrix[y, x, 1])

            if (lambda_max > 4 * lambda_min): line_structure_const[y, x] = 1
            else: line_structure_const[y, x] = 0

    line_structure_const = cv2.normalize(line_structure_const, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    court_line_candidate = cv2.normalize(court_line_candidate, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    line_structure_const_and = cv2.bitwise_and(court_line_candidate, court_line_candidate, mask=line_structure_const)

   
    ###################################
    # 3.2 Court Line Candidate Detector
    ###################################

    # canny edge -> hough transform by skimage
    # blur_canny = cv2.Canny(line_structure_const_and, 50, 200, None, 3)

    temp = img_as_float(line_structure_const_and)
    blur_canny = canny(temp, sigma=3)
    # print(blur_canny.shape)
    # print(blur_canny)

    # hough line transform from skimage
    tested_angles = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    h, theta, d = hough_line(blur_canny, theta=tested_angles)

    # print(h.shape)
    # print(theta.shape)
    # print(d.shape)

    img0 = np.array(img)

    thresh_hough = 0.42 * np.amax(h)

    for id, (_, angle, dist) in enumerate(zip(*hough_line_peaks(h, theta, d, threshold=thresh_hough))):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        slope = np.tan(angle + np.pi/2)

        # draw a line on opencv
        l = Line.from_point_slope(id, (x0, y0), slope)
        l.draw_line_extended(img0, (255, 0, 0))

    blur_canny = img_as_ubyte(blur_canny)

    while True:
        cv2.imshow('img', img)
        cv2.imshow('img_ycbcr', img_ycbcr)
        cv2.imshow('court_line_cand', court_line_candidate)
        cv2.imshow('line_struct_const', line_structure_const)
        cv2.imshow('line_struct_const_and', line_structure_const_and)
        cv2.imshow('blur_canny', blur_canny)
        cv2.imshow('img0', img0)

        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    cv2.destroyAllWindows()



def court_line_formula(img_y, y, x, tau, threshold_l, threshold_d):
    '''
    Forumla for determine whether a pixel is court line candidate
    in 3.1 White pixel detection
    '''
    # height, width, _ = img_y.shape
    # print("(x, y) = ({}, {})".format(x, y))

    if ((img_y[y, x] >= threshold_l) and (img_y[y, x] - img_y[y, x-tau] > threshold_d) and (img_y[y, x] - img_y[y, x+tau] > threshold_d)):
        return 1

    elif ((img_y[y, x] >= threshold_l) and (img_y[y, x] - img_y[y - tau, x] > threshold_d) and (img_y[y, x] - img_y[y+tau, x] > threshold_d)):
        return 1

    else: return 0

if __name__ == '__main__':
    tennis_court()