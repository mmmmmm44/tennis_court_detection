import cv2
import numpy as np
from pathlib import Path

# custom sub-classes
from tennis_court_model import TennisCourtModel
from utils import draw_court_model_to_img, draw_lines
from white_pixel_detector import WhitePixelDetector
from court_line_candidate_detector import CourtLineCandidateDetector
from model_fitting import ModelFitting

def main():
    # img_path = Path('test_images/tennis_pic_01.png')
    # img_path = Path('test_images/tennis_pic_02.png')
    # img_path = Path('test_images/tennis_pic_03.png')
    # img_path = Path('test_images/tennis_pic_04.png')
    # img_path = Path('test_images/tennis_pic_05.png')
    # img_path = Path('test_images/tennis_pic_06.png')
    # img_path = Path('test_images/tennis_pic_07.png')
    img_path = Path('test_images/tennis_pic_08.png')


    # tennis court image
    img = cv2.imread(str(img_path))
    
    img = resize_img(img)

    # Init detectors
    whitePixelDetector = WhitePixelDetector()
    courtLineCandidateDetector = CourtLineCandidateDetector()
    modelFitting = ModelFitting()
    
    best_model, score_max = court_model_init(img, whitePixelDetector, courtLineCandidateDetector, modelFitting)


    # visualization
    img_1 = np.array(img)
    draw_lines(img_1, modelFitting.lines_horizontal)
    draw_lines(img_1, modelFitting.lines_vertical)

    # Draw the projected court model to the image
    result_img = np.array(img)
    draw_court_model_to_img(result_img, best_model)

    print('Best model:')
    print(best_model)
    print("Best score:", score_max)

    while True:
        cv2.imshow('img', img)
        cv2.imshow('court_line_cand', whitePixelDetector.court_line_candidate)
        cv2.imshow('line_struct_const', whitePixelDetector.line_structure_const)
        cv2.imshow('line_struct_const_and', whitePixelDetector.line_structure_const_and)
        cv2.imshow('blur_canny', courtLineCandidateDetector.blur_canny)
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


def court_model_init(img, whitePixelDetector, courtLineCandidateDetector, modelFitting):
    ###################################
    # 3.1 White Pixel Detection
    ###################################
    
    line_structure_const_and = whitePixelDetector.execute(img)

    # court_line_candidate = whitePixelDetector.court_line_candidate
    # line_structure_const = whitePixelDetector.line_structure_const


    ###################################
    # 3.2 Court Line Candidate Detector
    ###################################

    lines_extended = courtLineCandidateDetector.execute(img, line_structure_const_and)

    # blur_canny = court_line_candidate_detector.blur_canny


    ###################################
    # 3.3 Model Fitting
    ###################################

    best_model, score_max = modelFitting.execute(img, lines_extended, line_structure_const_and)

    return best_model, score_max

def resize_img(img, target_h=960):
    height, width, _ = img.shape
    if height > target_h:
        w_h_ratio = width / float(height)
        img = cv2.resize(img, (int(target_h * w_h_ratio), target_h), interpolation=cv2.INTER_AREA)

    return img

if __name__ == '__main__':
    main()