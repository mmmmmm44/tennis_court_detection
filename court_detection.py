# trying to implement the algorithm in the paper
# Robust Camera Calibration for Sport Videos using Court Models

from collections import deque
from time import time
from unittest import result

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


def main_video():
    cap = cv2.VideoCapture('test_videos/Play 1.mp4')
    
    # TODO section 4

    if not cap.isOpened():
        print('Failed to open video file. Please check the file name.')

    # last two frames H for predicting t+1 frame
    past_H_deque = deque(maxlen=2)

    # Init detectors
    whitePixelDetector = WhitePixelDetector()
    courtLineCandidateDetector = CourtLineCandidateDetector()
    modelFitting = ModelFitting()

    # Init court model lines
    court_model_lines_h, court_model_lines_v = TennisCourtModel.court_model_lines_h, TennisCourtModel.court_model_lines_v
    court_model_lines = court_model_lines_h + court_model_lines_v


    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("End of video.")
            break

        # frame_count = cap.get(cv2.CAP_PROP_POS_FRAMES)
        # print('frame_count:', frame_count)

        # resize frame to reduce computation time
        frame = resize_img(frame)

        if len(past_H_deque) < past_H_deque.maxlen:
             # init court model
            best_model, score_max = court_model_init(frame, whitePixelDetector, courtLineCandidateDetector, modelFitting)

            result_img = np.array(frame)
            draw_court_model_to_img(result_img, best_model)

            # save the model to the queue
            past_H_deque.append(best_model)

            cv2.imshow('Frame', result_img)

        # with enough past data for estimation
        else:
            height, width = frame.shape[:2]

            # 3.1 only
            line_structure_const_and = whitePixelDetector.execute(frame)

            # Get M = H^{-1}_{t+1}, given H_{t-1}, H_{t}
            H_t_minus_1, H_t = tuple(past_H_deque)

            # @ = np.matmul()
            H_t_plus_1 = H_t @ np.linalg.inv(H_t_minus_1) @ H_t
            M = np.linalg.inv(H_t_plus_1)

            # project white pixels to court model
            # use pixel by pixel for loop to achieve that

            # possible improvement: multi-process with Queue as structure to handle the dictionary

            white_pixels_cords = []
            closest_model_lines = []

            for y in range(height):
                for x in range(width):
                    if line_structure_const_and[y, x] == 0: continue

                    # projection
                    projected_cord = np.matmul(M, np.array([x, y, 1]), dtype=np.float32)
                    projected_cord = projected_cord / projected_cord[2]

                    closest_dist, closest_line_id = float('inf'), -1

                    # find the cloest court model line
                    for line in court_model_lines:
                        dist = line.dist_btw_point((projected_cord[0], projected_cord[1]))
                        if dist < closest_dist:
                            closest_dist = dist
                            closest_line_id = line.id

                    # ignore pixels that the closest_dist is larger than a certain value
                    if closest_dist > 30:
                        continue


                    white_pixels_cords.append(np.array([x, y, 1]))
                    closest_model_lines.append(court_model_lines[closest_line_id].get_parameterized())

            white_pixels_cords = np.array(white_pixels_cords)       # shape = (N, 3)
            closest_model_lines = np.array(closest_model_lines)     # shape = (N, 3)
                    
                    
            # Python implementation of LM Algorithm, included example in solving transformations
            # https://github.com/jjhartmann/Levenberg-Marquardt-Algorithm
            # or using scipy.optimize.root() ??

            projection_error_function(M, (white_pixels_cords, closest_model_lines))


            cv2.imshow('Frame', frame)
        
        k = cv2.waitKey(0)
        # if k == ord('q'):
        #     break


# the projection error function D
def projection_error_function(params, args):
    '''
    params: numpy array [H11, H12, H13, H21, ..., H31, H32, H33]
    args: (white_pixels_cords, closest_model_lines)
    '''
    M = params

    white_pixels_cords, closest_model_lines = args

    print(white_pixels_cords.shape)
    print(white_pixels_cords)

    print(closest_model_lines.shape)
    print(closest_model_lines)

    D = 0

    # Do the cost function
    # Step 1: M @ white pixels

    white_pixels_cords = white_pixels_cords.transpose()     # shape = (3, N)

    np_Mp = M @ white_pixels_cords                        # (3, 3) @ (3, N) = (3, N)

    # normalize
    np_PMp = np_Mp / np_Mp[2, :]

    # multiply & sum to minick the dot and summation operation
    np_temp = np.einsum("ij,ji->ij", closest_model_lines, np_PMp)
    np_temp = np_temp.sum(axis=1)

    # square and get D
    D = np_temp @ np_temp
    
    return D


def resize_img(img, target_h=960):
    height, width, _ = img.shape
    if height > target_h:
        w_h_ratio = width / float(height)
        img = cv2.resize(img, (int(target_h * w_h_ratio), target_h), interpolation=cv2.INTER_AREA)

    return img


def draw_lines(img, lines, color=(255, 0, 0)):
    for line in lines:
        line.draw_line_extended(img, color)


def draw_court_model_to_img(img, H):

    _draw_court_model_lines_to_img(img, H, TennisCourtModel.court_model_lines_h)
    _draw_court_model_lines_to_img(img, H, TennisCourtModel.court_model_lines_v)

    return img

def _draw_court_model_lines_to_img(img, H, court_model_lines):
    for line in court_model_lines:
        start_pt_t = np.matmul(H, np.array([line.start_pt[0], line.start_pt[1], 1]))
        end_pt_t = np.matmul(H, np.array([line.end_pt[0], line.end_pt[1], 1]))
        start_pt_t = start_pt_t / start_pt_t[2]
        end_pt_t = end_pt_t / end_pt_t[2]

        mid_pt = (int((start_pt_t[0] + end_pt_t[0]) / 2), int((start_pt_t[1] + end_pt_t[1]) / 2))
        cv2.putText(img, str(line.id), mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.line(img, (int(start_pt_t[0]), int(start_pt_t[1])), (int(end_pt_t[0]), int(end_pt_t[1])), (255, 255, 0), 2)

if __name__ == '__main__':
    # main()
    main_video()
    # generate_court_model_lines()