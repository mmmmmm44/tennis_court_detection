# trying to implement the algorithm in the paper
# Robust Camera Calibration for Sport Videos using Court Models

from time import time

import cv2
import numpy as np


# custom sub-classes
from tennis_court_model import TennisCourtModel

from white_pixel_detector import WhitePixelDetector
from court_line_candidate_detector import CourtLineCandidateDetector
from model_fitting import ModelFitting


def main():
    # tennis court image
    img = cv2.imread('test_images/tennis_pic_02.png')
    height, width, _ = img.shape
    if height > 960:
        w_h_ratio = width / float(height)
        img = cv2.resize(img, (int(960 * w_h_ratio), 960), interpolation=cv2.INTER_AREA)
        height, width, _ = img.shape

    ###################################
    # 3.1 White Pixel Detection
    ###################################

    whitePixelDetector = WhitePixelDetector(img)
    
    line_structure_const_and = whitePixelDetector.execute()

    court_line_candidate = whitePixelDetector.court_line_candidate
    line_structure_const = whitePixelDetector.line_structure_const

    ###################################
    # 3.2 Court Line Candidate Detector
    ###################################

    court_line_candidate_detector = CourtLineCandidateDetector(img, line_structure_const_and)
    lines_extended = court_line_candidate_detector.execute()

    blur_canny = court_line_candidate_detector.blur_canny


    ###################################
    # 3.3 Model Fitting
    ###################################

    # get court model line parameters
    court_model_h, court_model_v, court_model_lines_h, court_model_lines_v = TennisCourtModel.y, TennisCourtModel.x, TennisCourtModel.court_model_lines_h, TennisCourtModel.court_model_lines_v

    model_fitting = ModelFitting(lines_extended, img, line_structure_const_and)
    best_model, score_max = model_fitting.execute()

    # visualization
    img_1 = np.array(img)
    draw_lines(img_1, model_fitting.lines_horizontal)
    draw_lines(img_1, model_fitting.lines_vertical)

    # Draw the projected court model to the image
    result_img = np.array(img)
    draw_court_model_to_img(result_img, best_model, court_model_lines_h)
    draw_court_model_to_img(result_img, best_model, court_model_lines_v)

    print('Best model:')
    print(best_model)
    print("Best score:", score_max)

    while True:
        cv2.imshow('img', img)
        cv2.imshow('court_line_cand', court_line_candidate)
        cv2.imshow('line_struct_const', line_structure_const)
        cv2.imshow('line_struct_const_and', line_structure_const_and)
        cv2.imshow('blur_canny', blur_canny)
        cv2.imshow('lines extended result', img_1)
        # cv2.imshow('lines extended horizontal', img_l_h)
        # cv2.imshow('lines extended vertical', img_l_v)
        cv2.imshow('result', result_img)
        # cv2.imshow('line 1', img_line_1)
        # cv2.imshow('sobel line', grad)

        # for k, _img in court_line_cand_img.items():
        #     cv2.imshow('court line {}'.format(k), _img)
        
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

def main_video():
    video = cv2.VideoCapture('test_videos/abc.mp4')
    
    # TODO section 4


def draw_lines(img, lines, color=(255, 0, 0)):
    for line in lines:
        line.draw_line_extended(img, color)

def draw_court_model_to_img(img, H, court_model_lines):
    for line in court_model_lines:
        start_pt_t = np.matmul(H, np.array([line.start_pt[0], line.start_pt[1], 1]))
        end_pt_t = np.matmul(H, np.array([line.end_pt[0], line.end_pt[1], 1]))
        start_pt_t = start_pt_t / start_pt_t[2]
        end_pt_t = end_pt_t / end_pt_t[2]

        mid_pt = (int((start_pt_t[0] + end_pt_t[0]) / 2), int((start_pt_t[1] + end_pt_t[1]) / 2))
        cv2.putText(img, str(line.id), mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.line(img, (int(start_pt_t[0]), int(start_pt_t[1])), (int(end_pt_t[0]), int(end_pt_t[1])), (255, 255, 0), 2)

if __name__ == '__main__':
    main()
    # generate_court_model_lines()