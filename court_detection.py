# trying to implement the algorithm in the paper
# Robust Camera Calibration for Sport Videos using Court Models

import itertools
import math
import multiprocessing as mp

import cv2
import numpy as np

from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage import img_as_ubyte, img_as_float

from scipy import stats

# custom sub-classes
from line import Line
from tennis_court_model import TennisCourtModel

# global constants
from config import APERATURE_SIZE, CLS_ANGLE_THRESH, THRESHOLD_R, HOUGH_THRESHOLD, TAU, THRESHOLD_L, THRESHOLD_D, BLOCK_SIZE


def main():
    # tennis court image
    img = cv2.imread('test_images/tennis_pic_05.png')
    height, width, _ = img.shape
    if height > 960:
        w_h_ratio = width / float(height)
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

    tau = TAU
    threshold_l = THRESHOLD_L
    threshold_d = THRESHOLD_D

    # about 50% faster (3.3x sec -> 1.7x sec) than single process version
    court_line_candidate = get_court_line_image_mp(img_y_int32, height, width, tau, threshold_l, threshold_d)

    # exclude pixels that are in textured regions
    
    # previously implemented CourtLinePixelDetector::computeStructureTensorElements from https://github.com/gchlebus/tennis-court-detection
    # result: not as good as mine

    block_size = BLOCK_SIZE             # affect the most in edge/ surface detection
    aperture_size = APERATURE_SIZE
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

    # first gaussian blur
    # then canny edge
    # then HoughTransform (or Hough Transform P)
    img_0 = np.array(img)
    img_1 = np.array(img)
    img_2 = np.array(img)

    lines_extended = {}

    # using hough transform from skimage
    temp = img_as_float(line_structure_const_and)
    blur_canny = canny(temp, sigma=3)

    # hough line transform from skimage
    tested_angles = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    h, theta, d = hough_line(blur_canny, theta=tested_angles)

    thresh_hough = HOUGH_THRESHOLD * np.amax(h)

    for id, (_, angle, dist) in enumerate(zip(*hough_line_peaks(h, theta, d, threshold=thresh_hough))):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        slope = np.tan(angle + np.pi/2)

        # draw a line on opencv
        l = Line.from_point_slope(id, (x0, y0), slope)
        lines_extended[id] = l
        # l.draw_line_extended(img_1, (255, 0, 0))

    blur_canny = img_as_ubyte(blur_canny)


    #################### 
    # visualization
    #################### 

    # img_lines_only = np.zeros((height, width))
    # for key, line in lines_extended.items():
    #     start_pt, end_pt = line[:2], line[2:]
    #     mid_pt = (int((start_pt[0] + end_pt[0]) / 2), int((start_pt[1] + end_pt[1]) / 2))
    #     cv2.putText(img_1, str(key), mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
    #     cv2.line(img_1, (int(start_pt[0]), int(start_pt[1])), (int(end_pt[0]), int(end_pt[1])), (255, 0, 255), 2)
    #     cv2.line(img_lines_only, (int(start_pt[0]), int(start_pt[1])), (int(end_pt[0]), int(end_pt[1])), (255, 255, 255), 2)

    # for line in lines_extended.values():
    #     line.draw_line_extended(img_1, (255, 0, 255))


    # parameterized every lines by its normal (normalized, start coordinate is (0, 0))
    for line in lines_extended.values():
        line.line_parameterization()

        # test = np.array(img)
        # line.draw_line_extended(test, (255, 0, 0))
        # line.draw_normal(test, (255, 127, 127), length=80)
        # cv2.imshow('test', test)
        # cv2.waitKey(1)


    # the set L of court line candidate pixels for each line
    court_line_candidate_pixels = {}
    dist_thresh = THRESHOLD_R

    for y in range(line_structure_const_and.shape[0]):
        for x in range(line_structure_const_and.shape[1]):
            # check for zero pixels
            if line_structure_const_and[y, x] < 127:
                continue

            for line in lines_extended.values():
                p = np.array([x, y, 1])
                q = np.array(line.parameterized)
                if abs(np.dot(q, p)) < dist_thresh:

                    if line.id not in court_line_candidate_pixels:
                        court_line_candidate_pixels[line.id] = [[], []]

                    # court_line_candidate_pixels[line.id].append([x, y])
                    court_line_candidate_pixels[line.id][0].append(x)
                    court_line_candidate_pixels[line.id][1].append(y)


    # print(court_line_candidate_pixels.keys())


    #################### 
    # visualization
    #################### 
    court_line_cand_img = {}
    for k in court_line_candidate_pixels.keys():
        img_line = np.zeros((height, width))
        lists = court_line_candidate_pixels[k]
        for p in zip(lists[0], lists[1]):
            img_line[p[1], p[0]] = 1

        court_line_cand_img[k] = img_line


    # Apply LMedS estimator. Yet I cannot find any python implementation.
    # A similar one with same % breakdown point will be "Robust Regression Using Repeated Medians".
    # scipy.stats.siegelslopes implements the paper.

    refinded_lines = {}

    for key, points in court_line_candidate_pixels.items():
        x = points[0]
        y = points[1]
        m, c = stats.siegelslopes(y, x)

        regressed = Line.from_point_slope(id=key, p1=(0, c), m=m)
        
        # line paramaterization for the new line
        regressed.line_parameterization()

        refinded_lines[key] = regressed

        # visualization
        # img_siegelslope = np.array(img)
        # lines_extended[key].draw_line_extended(img_siegelslope, (255, 0, 0))
        # regressed.draw_line_extended(img_siegelslope, (0, 0, 255))
        # draw the normal
        # lines_extended[key].draw_normal(img_siegelslope, (255, 127, 127), length=80, thickness=2)
        # cv2.imshow('img_sigelslope', img_siegelslope)
        # cv2.imshow('line {}'.format(key), court_line_cand_img[key])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


    # replace the old one
    lines_extended = refinded_lines

    ###################################
    # 3.3.1 Finding Line Correspondences
    ###################################

    # create two sets, one is horizontal, one is vertical
    lines_horizontal, lines_vertical = [], []
    degree_h = CLS_ANGLE_THRESH
    degree_h_in_rad = degree_h * math.pi / 180

    for line in lines_extended.values():
        result_pyatan2 = math.atan2(line.m_y, line.m_x)
        if abs(result_pyatan2) < degree_h_in_rad:
            lines_horizontal.append(line)
        else:
            lines_vertical.append(line)

    img_l_h = np.array(img_1)
    img_l_v = np.array(img_1)

    ###############
    # Visualization
    ###############

    for line in lines_horizontal:
        line.draw_line_extended(img_l_h, (255, 255, 0))
        
    for line in lines_vertical:
        line.draw_line_extended(img_l_v, (255, 0, 0))

    # print('lines_horizontal:', lines_horizontal.items())
    # print('lines_vertical:', lines_vertical.items())

    # sort the lines
    def sort_dist_key(line_para, p1):
        return np.dot(line_para, p1)

    lines_horizontal.sort(
        key = lambda x: sort_dist_key(x.parameterized, [int(width / 2.0), 0, 1]), reverse=True
    )
    lines_vertical.sort(
        key = lambda x: sort_dist_key(x.parameterized, [0, int(height / 2.0), 1]), reverse=True
    )


    # Moved the check duplicates to 
    # after classifying the lines into two groups, horizontal and vertical
    last = unique(lines_horizontal)
    lines_horizontal = lines_horizontal[:last]
    
    last = unique(lines_vertical)
    lines_vertical = lines_vertical[:last]

    for line in lines_horizontal:
        line.draw_line_extended(img_0, (255, 255, 0))

    for line in lines_vertical:
        line.draw_line_extended(img_0, (255, 0, 0))

    print('Num of lines_horizontal:', len(lines_horizontal))
    print('Num of lines_vertical:', len(lines_vertical))

    # get court model line parameters
    court_model_h, court_model_v, court_model_lines_h, court_model_lines_v = TennisCourtModel.y, TennisCourtModel.x, TennisCourtModel.court_model_lines_h, TennisCourtModel.court_model_lines_v

    # "... determine the best line assignment by iterating through all possible assignments"

    saved_model = None

    compensate_matrix = np.array(
        [[1, 0, -width / 2.0],
         [0, 1, -height / 2.0],
         [0, 0, 1]]
    )

    court_line_candidate_invert = np.invert(court_line_candidate)

    score_max = float('-inf')

    for i in range(len(lines_horizontal)):
        for k in range(i + 1, len(lines_horizontal)):
            for m in range(len(lines_vertical)):
                for n in range(m + 1, len(lines_vertical)):
                    line_h_i = lines_horizontal[i]
                    line_h_k = lines_horizontal[k]
                    line_v_m = lines_vertical[m]
                    line_v_n = lines_vertical[n]

                    # use linear algebra instead of the cross product in the paper
                    p1 = Line.solve_intersection(line_h_i, line_v_m)
                    p2 = Line.solve_intersection(line_h_i, line_v_n)
                    p3 = Line.solve_intersection(line_h_k, line_v_m)
                    p4 = Line.solve_intersection(line_h_k, line_v_n)

                    # print('p1:', p1, ', p2:', p2, ', p3', p3, ', p4:', p4)
                    # print('p1_cross:', p1_cross, 'p2_cross:', p2_cross, 'p3_cross:', p3_cross, 'p4_cross:', p4_cross)

                    ###################################
                    # Debug
                    ###################################
                    # p1_draw = tuple(map(int, p1))
                    # p2_draw = tuple(map(int, p2))
                    # p3_draw = tuple(map(int, p3))
                    # p4_draw = tuple(map(int, p4))

                    # court_img_test = np.array(img)
                    # line_h_i.draw_line_extended(court_img_test, (255, 255, 0))
                    # line_h_k.draw_line_extended(court_img_test, (255, 255, 0))
                    # line_v_m.draw_line_extended(court_img_test, (255, 0, 0))
                    # line_v_n.draw_line_extended(court_img_test, (255, 0, 0))

                    # cv2.putText(court_img_test, 'p1', (p1_draw[0], p1_draw[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
                    # cv2.putText(court_img_test, 'p2', (p2_draw[0], p2_draw[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
                    # cv2.putText(court_img_test, 'p3', (p3_draw[0], p3_draw[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
                    # cv2.putText(court_img_test, 'p4', (p4_draw[0], p4_draw[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

                    # cv2.circle(court_img_test, p1_draw[:2], 5, (0, 255, 0), -1)
                    # cv2.circle(court_img_test, p2_draw[:2], 5, (0, 255, 85), -1)
                    # cv2.circle(court_img_test, p3_draw[:2], 5, (0, 255, 170), -1)
                    # cv2.circle(court_img_test, p4_draw[:2], 5, (0, 255, 255), -1)
                    # cv2.imshow('court_img_test', court_img_test)
                    # cv2.waitKey(0)

                    for ii in range(len(court_model_h)):
                        for kk in range(ii + 1, len(court_model_h)):
                            for mm in range(len(court_model_v)):
                                for nn in range(mm + 1, len(court_model_v)):
                            
                                    line_cm_v1 = court_model_v[mm]
                                    line_cm_v2 = court_model_v[nn]
                                    line_cm_h1 = court_model_h[ii]
                                    line_cm_h2 = court_model_h[kk]

                                    p_cm_1 = [line_cm_v1, line_cm_h1]
                                    p_cm_2 = [line_cm_v2, line_cm_h1]
                                    p_cm_3 = [line_cm_v1, line_cm_h2]
                                    p_cm_4 = [line_cm_v2, line_cm_h2]

                                    # print('line_h_i:', line_h_i.id, ' ; line_h_k:', line_h_k.id)
                                    # print('line_v_m:', line_v_m.id, ' ; line_v_n:', line_v_n.id)
                                    # print('p1:', p1, ', p2:', p2, ', p3', p3, ', p4:', p4)
                                    # print('p_cm_1:', p_cm_1, ', p_cm_2:', p_cm_2, ', p_cm_3:', p_cm_3, ', p_cm_4:', p_cm_4)

                                    pts_src, pts_dest = np.array([p_cm_1[:2], p_cm_2[:2], p_cm_3[:2], p_cm_4[:2]]), np.array([p1[:2], p2[:2], p3[:2], p4[:2]])
                                    # print(pts_src)
                                    # print(pts_dest)
                                    H, status = cv2.findHomography(pts_src, pts_dest, cv2.RANSAC, 5.0)

                                    ###################################
                                    # 3.3.2 Fast Calibration Parameter Rejection Test
                                    ###################################

                                    H_pi = np.matmul(
                                        compensate_matrix, np.array(H)
                                    )

                                    # get f^2 and beta^2 from adjusted Homographhy matrix
                                    f_square = -((H_pi[0,0] * H_pi[0,1] + H_pi[1,0] * H_pi[1,1]) / (H_pi[2,0] * H_pi[2,1]))
                                    beta_square = (H_pi[0,1] * H_pi[0,1] + H_pi[1,1] * H_pi[1,1] + f_square * H_pi[2,1] * H_pi[2,1]) / (H_pi[0,0] * H_pi[0,0] + H_pi[1,0] * H_pi[1,0] + f_square * H_pi[2,0] * H_pi[2,0])
                                    
                                    if ((beta_square < 0.5 * 0.5) or (beta_square > 2 * 2)):
                                        continue


                                    ###################################
                                    # 3.3.3 Evaluating the Model Support
                                    ###################################

                                    # transform all line segments of the model to the image

                                    score = 0

                                    court_transform = np.array(img)

                                    trans_court_model = np.zeros((height, width))
                                    
                                    for line_h in court_model_lines_h:
                                        start_pt_t = np.matmul(H, np.array([line_h.start_pt[0], line_h.start_pt[1], 1]))
                                        end_pt_t = np.matmul(H, np.array([line_h.end_pt[0], line_h.end_pt[1], 1]))
                                        start_pt_t = start_pt_t / start_pt_t[2]
                                        end_pt_t = end_pt_t / end_pt_t[2]

                                        cv2.line(trans_court_model, (int(start_pt_t[0]), int(start_pt_t[1])), (int(end_pt_t[0]), int(end_pt_t[1])), (255, 255, 255), 2)

                                        # Debug use
                                        mid_pt = (int((start_pt_t[0] + end_pt_t[0]) / 2), int((start_pt_t[1] + end_pt_t[1]) / 2))
                                        cv2.putText(court_transform, str(line_h.id), mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
                                        cv2.line(court_transform, (int(start_pt_t[0]), int(start_pt_t[1])), (int(end_pt_t[0]), int(end_pt_t[1])), (255, 255, 0), 2)

                                    for line_v in court_model_lines_v:
                                        start_pt_t = np.matmul(H, np.array([line_v.start_pt[0], line_v.start_pt[1], 1]))
                                        end_pt_t = np.matmul(H, np.array([line_v.end_pt[0], line_v.end_pt[1], 1]))
                                        start_pt_t = start_pt_t / start_pt_t[2]
                                        end_pt_t = end_pt_t / end_pt_t[2]

                                        cv2.line(trans_court_model, (int(start_pt_t[0]), int(start_pt_t[1])), (int(end_pt_t[0]), int(end_pt_t[1])), (255, 255, 255), 2)

                                        # Debug use
                                        mid_pt = (int((start_pt_t[0] + end_pt_t[0]) / 2), int((start_pt_t[1] + end_pt_t[1]) / 2))
                                        cv2.putText(court_transform, str(line_v.id), mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
                                        cv2.line(court_transform, (int(start_pt_t[0]), int(start_pt_t[1])), (int(end_pt_t[0]), int(end_pt_t[1])), (255, 0, 0), 2)


                                    trans_court_model = cv2.normalize(trans_court_model, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                                    # if l(x, y) -> 1
                                    on_court = cv2.bitwise_and(trans_court_model, trans_court_model, mask=court_line_candidate)
                                    # if not in l(x,y) -> -1/2
                                    not_on_court = cv2.bitwise_and(trans_court_model, trans_court_model, mask=court_line_candidate_invert)
                                    # else -> 0 (do nth)

                                    score += np.count_nonzero(on_court)
                                    score += np.count_nonzero(not_on_court) * -0.5

                                    if score > score_max:
                                        score_max = score
                                        saved_model = H

                                        ###############
                                        # Debug
                                        ###############
                                        # cv2.imshow('court transform img', court_transform)
                                        # cv2.imshow('trans court model', trans_court_model)
                                        # cv2.imshow('on court', on_court)
                                        # cv2.imshow('not on court', not_on_court)
                                        # print('line_h_i:', line_h_i.id, ' ; line_h_k:', line_h_k.id)
                                        # print('line_v_m:', line_v_m.id, ' ; line_v_n:', line_v_n.id)
                                        # print('p1:', p1, ', p2:', p2, ', p3', p3, ', p4:', p4)
                                        # print('p_cm_1:', p_cm_1, ', p_cm_2:', p_cm_2, ', p_cm_3:', p_cm_3, ', p_cm_4:', p_cm_4)
                                        # print('H =', H)
                                        # print('score = {}'.format(score))
                                        # cv2.imshow('court transform test', trans_court_model)
                                        # cv2.waitKey(0)

                                    ###################################
                                    # Debug
                                    ###################################
                                    # court_model_img_test = np.array(court_model_img)
                                    # court_img_test = np.array(img)
                                    # line_h_i.draw_line_extended(court_img_test, (255, 0, 0))
                                    # line_h_k.draw_line_extended(court_img_test, (255, 0, 0))
                                    # line_v_m.draw_line_extended(court_img_test, (255, 0, 0))
                                    # line_v_n.draw_line_extended(court_img_test, (255, 0, 0))
                                    # cv2.circle(court_img_test, tuple(map(int, p1))[:2], 5, (0, 255, 0), -1)
                                    # cv2.circle(court_img_test, tuple(map(int, p2))[:2], 5, (0, 255, 85), -1)
                                    # cv2.circle(court_img_test, tuple(map(int, p3))[:2], 5, (0, 255, 170), -1)
                                    # cv2.circle(court_img_test, tuple(map(int, p4))[:2], 5, (0, 255, 255), -1)
                                    # cv2.circle(court_model_img_test, tuple(map(int, p_cm_1))[:2], 2, (0, 255, 0), -1)
                                    # cv2.circle(court_model_img_test, tuple(map(int, p_cm_2))[:2], 2, (0, 255, 85), -1)
                                    # cv2.circle(court_model_img_test, tuple(map(int, p_cm_3))[:2], 2, (0, 255, 170), -1)
                                    # cv2.circle(court_model_img_test, tuple(map(int, p_cm_4))[:2], 2, (0, 255, 255), -1)

                                    # cv2.imshow('court_model', court_model_img_test)
                                    # cv2.imshow('court_img_test', court_img_test)
                                    # cv2.waitKey(0)



    # Draw the projected court model to the image
    draw_court_model_to_img(img_2, saved_model, court_model_lines_h)
    draw_court_model_to_img(img_2, saved_model, court_model_lines_v)

    print('Best model:')
    print(H)
    print("Best score:", score_max)

    while True:
        cv2.imshow('img', img)
        cv2.imshow('court_line_cand', court_line_candidate)
        cv2.imshow('line_struct_const', line_structure_const)
        cv2.imshow('line_struct_const_and', line_structure_const_and)
        cv2.imshow('blur_canny', blur_canny)
        cv2.imshow('duplicates removed', img_0)
        # cv2.imshow('lines extended result', img_1)
        # cv2.imshow('lines extended horizontal', img_l_h)
        # cv2.imshow('lines extended vertical', img_l_v)
        cv2.imshow('result', img_2)
        # cv2.imshow('line 1', img_line_1)
        # cv2.imshow('sobel line', grad)

        # for k, _img in court_line_cand_img.items():
        #     cv2.imshow('court line {}'.format(k), _img)
        
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

def get_court_line_image_mp(int_y_int32, height, width, tau, threshold_l, threshold_d):
    '''
    Multi-process version to get court line pixels
    In 3.1 White pixel detection
    '''

    # reference from https://stackoverflow.com/questions/50602137/python-multithread-process-to-fill-in-matrix-with-values

    # construct pool
    pool = mp.Pool(mp.cpu_count())

    # organize the args
    args = [(int_y_int32, row, col, tau, threshold_l, threshold_d, height, width) for (row, col) in list(itertools.product(range(height), range(width)))]
    court_line_candidate = np.zeros((height, width))
    results = pool.starmap(court_line_formula, args)

    # unpack the results into the matrix
    for i_tuple, result in zip([(row, col) for (row, col) in list(itertools.product(range(height), range(width)))], results):
        # unpack
        r, c = i_tuple

        # set it in the matrix
        court_line_candidate[r, c] = result

    return court_line_candidate


def court_line_formula(img_y, y, x, tau, threshold_l, threshold_d, height, width):
    '''
    Forumla for determine whether a pixel is court line candidate
    in 3.1 White pixel detection
    '''

    if (x < tau) or (x >= width - tau):
        return 0

    if (y < tau) or (y >= height - tau):
        return 0

    if ((img_y[y, x] >= threshold_l) and (img_y[y, x] - img_y[y, x-tau] > threshold_d) and (img_y[y, x] - img_y[y, x+tau] > threshold_d)):
        return 1

    elif ((img_y[y, x] >= threshold_l) and (img_y[y, x] - img_y[y - tau, x] > threshold_d) and (img_y[y, x] - img_y[y+tau, x] > threshold_d)):
        return 1

    else: return 0


def unique(lines):
    '''
    remove duplicates in sorted list
    implementing std::unique(first, end) in c++
    '''

    if len(lines) == 1:
        return len(lines)

    first = 0
    last = len(lines)

    result = first
    
    while True:
        first += 1
        if first == last:
            break

        if (not (lines[result] == lines[first])):
            result += 1
            if (result != first):
                lines[result] = lines[first]

    return (result + 1)


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