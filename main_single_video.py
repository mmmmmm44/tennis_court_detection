# trying to implement the algorithm in the paper
# Robust Camera Calibration for Sport Videos using Court Models

from collections import deque
import os
from pathlib import Path
from unittest import result

import cv2
import numpy as np
from LA_Algorithm.LMA import LM

# custom sub-classes
from tennis_court_model import TennisCourtModel
from utils import draw_court_model_to_img, load_H_matrix, resize_img, save_H_matrix

from white_pixel_detector import WhitePixelDetector
from court_line_candidate_detector import CourtLineCandidateDetector
from model_fitting import ModelFitting


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

def show_video_with_projection_matrix():
    play_name = 'play_05'

    video_file_path = Path(f'test_videos/{play_name}.mp4')
    numpy_H_matrices_folder_path = Path(f"test_section_4_1_data/{play_name}")

    cap = cv2.VideoCapture(str(video_file_path))

    if not numpy_H_matrices_folder_path.exists():
        print('No pre-calculated result is shown. Program terminates.')
        return

    no_of_results = len(next(os.walk(numpy_H_matrices_folder_path))[2])

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f'total num of results: {no_of_results}')
    print(f'total num of frames: {length}')

    # if no_of_results != length:
    #     print('Number of pre-calculated results does not match with the total number of frames of the video. Program terminates')
    #     return
    
    frame_no = 1

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("End of video.")
            break

        # resize frame to reduce computation time
        frame = resize_img(frame)

        numpy_H_matrix_filename = Path(numpy_H_matrices_folder_path, f"H_matrix_frame_{frame_no}.npy")

        if not numpy_H_matrix_filename.exists():
            print(f'The precalculated numpy result for frame {frame_no} does not exist. Program terminates.')
            break

        H = load_H_matrix(numpy_H_matrix_filename)

        # project court model to the image
        draw_court_model_to_img(frame, H)

        cv2.imshow('frame', frame)
        k = cv2.waitKey(50)
        if k == ord('q'):
            break

        frame_no += 1

    cap.release()
    cv2.destroyAllWindows()

def main_video():
    play_name = 'play_05'

    video_file_path = Path(f'test_videos/{play_name}.mp4')
    numpy_H_matrices_folder_path = Path(f"test_section_4_1_data/{play_name}")

    cap = cv2.VideoCapture(str(video_file_path))
    
    if not cap.isOpened():
        print('Failed to open video file. Please check the file name.')

    if not numpy_H_matrices_folder_path.exists():
        numpy_H_matrices_folder_path.mkdir(parents=True)

    # last two frames H for predicting t+1 frame
    past_H_deque = deque(maxlen=2)

    # Init detectors
    whitePixelDetector = WhitePixelDetector()
    courtLineCandidateDetector = CourtLineCandidateDetector()
    modelFitting = ModelFitting()

    # Init court model lines
    court_model_lines_h, court_model_lines_v = TennisCourtModel.court_model_lines_h, TennisCourtModel.court_model_lines_v
    court_model_lines = court_model_lines_h + court_model_lines_v

    frame_no = 1

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("End of video.")
            break

        # frame_count = cap.get(cv2.CAP_PROP_POS_FRAMES)
        # print('frame_count:', frame_count)

        # resize frame to reduce computation time
        frame = resize_img(frame)


        numpy_H_matrix_filename = Path(numpy_H_matrices_folder_path, f"H_matrix_frame_{frame_no}.npy")

        if len(past_H_deque) < past_H_deque.maxlen:

            if numpy_H_matrix_filename.exists():
                best_model = load_H_matrix(numpy_H_matrix_filename)
            else: 
                # init court model
                best_model, score_max = court_model_init(frame, whitePixelDetector, courtLineCandidateDetector, modelFitting)

            result_img = np.array(frame)
            draw_court_model_to_img(result_img, best_model)

            # save the model to the queue
            past_H_deque.append(best_model)

            print(f'H frame_{frame_no}')
            print(best_model)

            # save numpy matrices for reducing computation time
            if not numpy_H_matrix_filename.exists():
                save_H_matrix(numpy_H_matrix_filename, best_model)

            cv2.imshow('Frame', result_img)

        # with enough past data for estimation
        else:
            # load from pre-calculated result
            if numpy_H_matrix_filename.exists():
                H_t_plus_1_LM = load_H_matrix(numpy_H_matrix_filename)
                print(f'Loaded precalculated frame {frame_no} result.')

                # save the model to the queue
                past_H_deque.append(H_t_plus_1_LM)

                frame_no += 1

                continue


            height, width = frame.shape[:2]

            # 3.1 only
            line_structure_const_and = whitePixelDetector.execute(frame)

            # Get M = H^{-1}_{t+1}, given H_{t-1}, H_{t}
            H_t_minus_1, H_t = tuple(past_H_deque)

            # @ = np.matmul()
            H_t_plus_1 = H_t @ np.linalg.inv(H_t_minus_1) @ H_t
            M = np.linalg.inv(H_t_plus_1)                           # M = an estimate of (H_t_plus_1)^(-1)

            # Debug
            # try project the court model to the image:
            # cv2.imshow('line_structure_const_and', line_structure_const_and)
            # cv2.waitKey(0)

            # project white pixels to court model
            # use pixel by pixel for loop to achieve that

            # possible improvement: multi-process with Queue as structure to handle the dictionary

            remaining_white_pixels = np.zeros((height, width), dtype=np.uint8)

            white_pixels_cords = []
            closest_model_lines = []

            for y in range(height):
                for x in range(width):
                    if line_structure_const_and[y, x] == 0: continue

                    # projecting classified white pixels to the court model
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
                    # the value is small since the declared court model is not large
                    if closest_dist > 1:
                        continue


                    remaining_white_pixels[y, x] = 255
                    white_pixels_cords.append(np.array([x, y, 1]))
                    closest_model_lines.append(court_model_lines[closest_line_id].get_parameterized())

            white_pixels_cords = np.array(white_pixels_cords)       # shape = (N, 3)
            closest_model_lines = np.array(closest_model_lines)     # shape = (N, 3)

            # cv2.imshow('Line struct const', line_structure_const_and)
            print('M:', M)
            # original_error = projection_error_function(M.reshape((9, 1)), (white_pixels_cords, closest_model_lines))
            # print('Original estimate error:', original_error, '; rms error:', np.square(original_error).sum())
                    
            cv2.imshow('remaining_white_pixels', remaining_white_pixels)
            # cv2.waitKey(1)

            # Python implementation of LM Algorithm, included example in solving transformations
            # https://github.com/jjhartmann/Levenberg-Marquardt-Algorithm
            # or using scipy.optimize.root() ??

            # Still don't know how to write the correct one...

            out = LM(
                seed_params=M.reshape((9,)),
                args=(white_pixels_cords, closest_model_lines),
                error_function=projection_error_function,
                lambda_multiplier=10,
                kmax=50,
                eps=0.001,
                verbose=True
            )

            if out == -1:
                print("Error occurred. The program will shut itself.")
                exit(0)

            rmserror, reason = out[0], out[2]

            print('\n\n')
            print('LM ends')
            print(f'rmserror: {rmserror}; reason: {reason}')
            
            M_star = out[1].reshape((3, 3))
            

            print('\n\n')
            print("Before LM")
            print(M)
            print("After LM")
            print("M_star:")
            print(M_star)

            # H_t_plus_1_LM = np.linalg.inv(M_star.reshape((3, 3)))
            H_t_plus_1_LM = np.linalg.inv(M_star.reshape((3, 3)))
            # Normalization
            H_t_plus_1_LM /= H_t_plus_1_LM[2, 2]

            print('Estimated M, to project court model to image. = inversed M_star')
            print(H_t_plus_1_LM)

            # Do the projection
            result_img = np.array(frame)
            draw_court_model_to_img(result_img, H_t_plus_1_LM)

            result_img2 = np.array(frame)
            draw_court_model_to_img(result_img2, H_t_plus_1)

            # save the model to the queue
            past_H_deque.append(H_t_plus_1_LM)

            print(f'H_est frame_{frame_no}')
            print(H_t_plus_1)

            cv2.imshow('Frame H_t_plus_1', result_img2)
            cv2.imshow('Frame H_t_plus_1_LM', result_img)


            # save numpy matrices for reducing computation time
            if not numpy_H_matrix_filename.exists():
                save_H_matrix(numpy_H_matrix_filename, H_t_plus_1_LM)
                print(f'Saved frame {frame_no} result after LM.')
        
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

        frame_no += 1

    cap.release()
    cv2.destroyAllWindows()

# the projection error function D
def projection_error_function(params, args):
    '''
    Calculate the error of projecting the white pixels to the court model.
    Error is defined as euclidian distance

    params: numpy array [H11, H12, H13, H21, ..., H31, H32, H33]
    args: (white_pixels_cords, closest_model_lines)
    '''
    M = params.reshape(3, 3)                            # checked

    white_pixels_cords, closest_model_lines = args      # shape: (N, 3); (N, 3)     # checked

    # print(white_pixels_cords.shape)
    # print(white_pixels_cords)
    # print(closest_model_lines.shape)
    # print(closest_model_lines)

    # Do the cost function
    # Step 1: M @ white pixels

    white_pixels_cords = white_pixels_cords.transpose()     # shape = (3, N)        # checked

    np_Mp = M @ white_pixels_cords                          # (3, 3) @ (3, N) = (3, N)

    # normalize
    np_PMp = np_Mp / np_Mp[2, :]                            # checked

    # multiply & sum to minick the dot and summation operation
    np_temp = np.einsum("ij,ji->i", closest_model_lines, np_PMp)

    # square each element in the 1D matrix, then add them -> gives D (in the paper)
    # np_temp = np_temp @ np_temp

    # does this as in the LM func instead of above
    # as they will apply np.norm() to this value to get the rms error
    # which is equivalent to squaring each element then sum all of them.
    np_temp = np_temp * np_temp

    return np_temp

if __name__ == '__main__':
    # main_video()
    show_video_with_projection_matrix()