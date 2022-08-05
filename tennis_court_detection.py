# trying to implement the algorithm in the paper
# Robust Camera Calibration for Sport Videos using Court Models

import cv2
from line import Line
from merge_line_banderlog013_maxi import HoughBundler
import numpy as np

from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage import img_as_ubyte, img_as_float

from scipy import stats

import math

from itertools import combinations

def generate_court_model_lines():

    # tennis court model and their lines (pre-defined)
    court_model_img = cv2.imread('tennis_court_clean.png')
    # court_model_img_1 = cv2.imread('tennis_court_clean.png')

    court_m_h, court_m_w, _ = court_model_img.shape
    y = [84, 274, 724, 915]          # y = a
    # court_model_lines_h_eqt = [84, 274, 500, 724, 915]   
    x = [131, 184, 330, 476, 529]         # x = b
    # court_model_lines_v_eqt = [131, 184, 330, 476, 529]  

    # forming equations from given points
    court_model_lines_h = [
        Line.from_two_point(0, (x[0], y[0]), (x[4], y[0])),
        Line.from_two_point(1, (x[1], y[1]), (x[3], y[1])),
        Line.from_two_point(2, (x[1], y[2]), (x[3], y[2])),
        Line.from_two_point(3, (x[0], y[3]), (x[4], y[3]))
    ]

    court_model_lines_v = [
        Line.from_two_point(4, (x[0], y[0]),(x[0], y[3])),
        Line.from_two_point(5, (x[1], y[0]),(x[1], y[3])),
        Line.from_two_point(6, (x[2], y[1]),(x[2], y[2])),
        Line.from_two_point(7, (x[3], y[0]),(x[3], y[3])),
        Line.from_two_point(8, (x[4], y[0]),(x[4], y[3])),
    ]

    #################### 
    # visualization
    ####################
    # for line in court_model_lines_h:
    #     start_pt, end_pt = line[:2], line[2:]
    #     cv2.line(court_model_img, (int(start_pt[0]), int(start_pt[1])), (int(end_pt[0]), int(end_pt[1])), (255, 0, 255), 2)
    # for line in court_model_lines_v:
    #     start_pt, end_pt = line[:2], line[2:]
    #     cv2.line(court_model_img, (int(start_pt[0]), int(start_pt[1])), (int(end_pt[0]), int(end_pt[1])), (255, 0, 255), 2)


    # while True:
    #     cv2.imshow('court_model_img_1', court_model_img_1)

    #     k = cv2.waitKey(1)
    #     if k == ord('q'):
    #         break

    return court_model_img, y, x, court_model_lines_h, court_model_lines_v



def main():
    # tennis court image
    img = cv2.imread('tennis_pic_06.png')
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

    block_size = 21         # affect the most in edge/ surface detection
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

    # first gaussian blur
    # then canny edge
    # then HoughTransform (or Hough Transform P)
    img_0 = np.array(img)
    img_1 = np.array(img)
    img_2 = np.array(img)

    # blur_canny = cv2.Canny(line_structure_const_and, 50, 200, None, 3)

    # linesP = cv2.HoughLinesP(blur_canny, 1, np.pi / 180, 50, None, 70, 10)
    
    # if linesP is not None:
    #     for i in range(0, len(linesP)):
    #         l = linesP[i][0]
    #         cv2.line(img_0, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

    # # unwrap the linesP array
    # lines = [line_i for line_i in [l[0] for l in linesP]]

    # # pre-process the detected lines (filtering out them)
    # bundler = HoughBundler(min_distance=30, min_angle=5)
    # lines_processed = bundler.process_lines(lines)
    # lines = lines_processed

    # print("number of remaining lines:", len(lines))

    # # lines_extended = {}

    # # extend the lines to cover the whole image
    # lines_extended = []

    # for key, line in enumerate(lines):
    #     l_obj = Line.from_two_point(key, p1=line[:2], p2=line[2:])

    #     lines_extended.append(l_obj)

    lines_extended = {}

    # using hough transform from skimage
    temp = img_as_float(line_structure_const_and)
    blur_canny = canny(temp, sigma=3)

    # hough line transform from skimage
    tested_angles = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    h, theta, d = hough_line(blur_canny, theta=tested_angles)

    thresh_hough = 0.42 * np.amax(h)

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
    dist_thresh = 6

    for y in range(line_structure_const_and.shape[0]):
        for x in range(line_structure_const_and.shape[1]):
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

        # visualization
        # img_siegelslope = np.array(img)
        regressed = Line.from_point_slope(id=key, p1=(0, c), m=m)
        
        # line paramaterization for the new line
        regressed.line_parameterization()

        refinded_lines[key] = regressed
        
        # cv2.imshow('img_sigelslope', img_siegelslope)
        # lines_extended[key].draw_line_extended(img_siegelslope, (255, 0, 0))
        # regressed.draw_line_extended(img_siegelslope, (0, 0, 255))
        # draw the normal
        # lines_extended[key].draw_normal(img_siegelslope, (255, 127, 127), length=80, thickness=2)
        # cv2.imshow('line {}'.format(key), court_line_cand_img[key])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


    # replace the old one
    lines_extended = refinded_lines

    # scanned for duplicates
    # but no duplicates XDD
    rad_thresh = math.cos(0.75 * math.pi / 180)
    dist_thresh = 1.5
    lines_para_dict_new = {}

    # print("Before scanned duplicates:", lines_parameterized.keys())

    while True:
        for (line_a, line_b) in combinations(lines_extended.values(), 2):
            if ((np.dot(line_a.parameterized[:2], line_b.parameterized[:2]) > rad_thresh) and (abs(-line_a.parameterized[2] + line_b.parameterized[2]) < 1.5)):
                # duplicate
                if line_a.id in lines_para_dict_new:
                    continue
                elif line_b.id in lines_para_dict_new:
                    continue
                else:
                    # add one only
                    lines_para_dict_new[line_a.id] = line_a 
            else:
                # add both
                if line_a.id not in lines_para_dict_new:
                    lines_para_dict_new[line_a.id] = line_a
                
                if line_b.id not in lines_para_dict_new:
                    lines_para_dict_new[line_b.id] = line_b


        # print("lines_parameterized keys:", lines_parameterized.keys())
        # print("lines_para_dict_new keys:", lines_para_dict_new.keys())
        if len(lines_para_dict_new) == len(lines_extended):
            break
        else:
            lines_extended = lines_para_dict_new
            lines_para_dict_new = {}


    # print("After scanned duplicates:", lines_extended)

    for line in lines_extended.values():
        line.draw_line_extended(img_0, (255, 0, 0))

    ###################################
    # 3.3.1 Finding Line Correspondences
    ###################################

    # create two sets, one is horizontal, one is vertical
    lines_horizontal, lines_vertical = [], []
    degree_h = 25
    degree_h_in_rad = degree_h * math.pi / 180

    for line in lines_extended.items():
        result_pyatan2 = math.atan2(line.m_y, line.m_x)
        if abs(result_pyatan2) < degree_h_in_rad:
            lines_horizontal.append(line)
        else:
            lines_vertical.append(line)

    img_l_h = np.array(img_1)
    img_l_v = np.array(img_1)

    for line in lines_horizontal:
        line.draw_line_extended(img_l_h, (255, 255, 0))
        
    for line in lines_vertical:
        line.draw_line_extended(img_l_v, (255, 0, 0))

    # print('lines_horizontal:', lines_horizontal.items())
    # print('lines_vertical:', lines_vertical.items())

    # sort the lines
    def sort_dist_key(line_para, p1):
        return np.dot(line_para, p1)

    # lines_h = list(lines_horizontal.values())
    # lines_v = list(lines_vertical.values())
    # lines_h.sort(key = lambda x: sort_dist_key(x, [int(width / 2.0), 0, 1]))
    # lines_v.sort(key = lambda x: sort_dist_key(x, [0, int(height / 2.0), 1]))

    # lines_h = {
    #     k: l for k, l in sorted(lines_horizontal.items(), key=lambda x: sort_dist_key(x[1], [int(width / 2.0), 0, 1]))
    # }
    # lines_v = {
    #     k: l for k, l in sorted(lines_vertical.items(), key=lambda x: sort_dist_key(x[1], [0, int(height / 2.0), 1]))
    # }

    lines_horizontal.sort(
        key = lambda x: sort_dist_key(x.parameterized, [int(width / 2.0), 0, 1]), reverse=True
    )
    lines_vertical.sort(
        key = lambda x: sort_dist_key(x.parameterized, [0, int(height / 2.0), 1]), reverse=True
    )

    # print('lines_horizontal:', lines_horizontal)
    # print('lines_vertical:', lines_vertical)

    # get court model line parameters
    court_model_img, court_model_h, court_model_v, court_model_lines_h, court_model_lines_v = generate_court_model_lines()

    # "... determine the best line assignment by iterating through all possible assignments"

    saved_model = None

    compensate_matrix = np.array(
        [[1, 0, -width / 2.0],
         [0, 1, -height / 2.0],
         [0, 0, 1]]
    )

    court_line_candidate_invert = np.invert(court_line_candidate)

    score_max = float('-inf')

    # for i in range(len(lines_horizontal)):
    #     for k in range(i + 1, len(lines_horizontal)):
    #         for m in range(len(lines_vertical)):
    #             for n in range(m + 1, len(lines_vertical)):
    #                 line_h_i = lines_horizontal[i]
    #                 line_h_k = lines_horizontal[k]
    #                 line_v_m = lines_vertical[m]
    #                 line_v_n = lines_vertical[n]

    #                 # use linear algebra instead of the cross product in the paper
    #                 p1 = Line.solve_intersection(line_h_i, line_v_m)
    #                 p2 = Line.solve_intersection(line_h_i, line_v_n)
    #                 p3 = Line.solve_intersection(line_h_k, line_v_m)
    #                 p4 = Line.solve_intersection(line_h_k, line_v_n)

    #                 # print('p1:', p1, ', p2:', p2, ', p3', p3, ', p4:', p4)
    #                 # print('p1_cross:', p1_cross, 'p2_cross:', p2_cross, 'p3_cross:', p3_cross, 'p4_cross:', p4_cross)

    #                 ###################################
    #                 # Debug
    #                 ###################################
    #                 # p1_draw = tuple(map(int, p1))
    #                 # p2_draw = tuple(map(int, p2))
    #                 # p3_draw = tuple(map(int, p3))
    #                 # p4_draw = tuple(map(int, p4))

    #                 # court_img_test = np.array(img)
    #                 # line_h_i.draw_line_extended(court_img_test, (255, 255, 0))
    #                 # line_h_k.draw_line_extended(court_img_test, (255, 255, 0))
    #                 # line_v_m.draw_line_extended(court_img_test, (255, 0, 0))
    #                 # line_v_n.draw_line_extended(court_img_test, (255, 0, 0))

    #                 # cv2.putText(court_img_test, 'p1', (p1_draw[0], p1_draw[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
    #                 # cv2.putText(court_img_test, 'p2', (p2_draw[0], p2_draw[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
    #                 # cv2.putText(court_img_test, 'p3', (p3_draw[0], p3_draw[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
    #                 # cv2.putText(court_img_test, 'p4', (p4_draw[0], p4_draw[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

    #                 # cv2.circle(court_img_test, p1_draw[:2], 5, (0, 255, 0), -1)
    #                 # cv2.circle(court_img_test, p2_draw[:2], 5, (0, 255, 85), -1)
    #                 # cv2.circle(court_img_test, p3_draw[:2], 5, (0, 255, 170), -1)
    #                 # cv2.circle(court_img_test, p4_draw[:2], 5, (0, 255, 255), -1)
    #                 # cv2.imshow('court_img_test', court_img_test)
    #                 # cv2.waitKey(0)

    #                 for ii in range(len(court_model_h)):
    #                     for kk in range(ii + 1, len(court_model_h)):
    #                         for mm in range(len(court_model_v)):
    #                             for nn in range(mm + 1, len(court_model_v)):
                            
    #                                 line_cm_v1 = court_model_v[mm]
    #                                 line_cm_v2 = court_model_v[nn]
    #                                 line_cm_h1 = court_model_h[ii]
    #                                 line_cm_h2 = court_model_h[kk]

    #                                 p_cm_1 = [line_cm_v1, line_cm_h1]
    #                                 p_cm_2 = [line_cm_v2, line_cm_h1]
    #                                 p_cm_3 = [line_cm_v1, line_cm_h2]
    #                                 p_cm_4 = [line_cm_v2, line_cm_h2]

    #                                 print('line_h_i:', line_h_i.id, ' ; line_h_k:', line_h_k.id)
    #                                 print('line_v_m:', line_v_m.id, ' ; line_v_n:', line_v_n.id)
    #                                 print('p1:', p1, ', p2:', p2, ', p3', p3, ', p4:', p4)
    #                                 print('p_cm_1:', p_cm_1, ', p_cm_2:', p_cm_2, ', p_cm_3:', p_cm_3, ', p_cm_4:', p_cm_4)

    #                                 pts_src, pts_dest = np.array([p_cm_1[:2], p_cm_2[:2], p_cm_3[:2], p_cm_4[:2]]), np.array([p1[:2], p2[:2], p3[:2], p4[:2]])
    #                                 print(pts_src)
    #                                 print(pts_dest)
    #                                 H, status = cv2.findHomography(pts_src, pts_dest, cv2.RANSAC, 5.0)

    #                                 ###################################
    #                                 # 3.3.2 Fast Calibration Parameter Rejection Test
    #                                 ###################################

    #                                 H_pi = np.matmul(
    #                                     compensate_matrix, np.array(H)
    #                                 )

    #                                 # get f and beta from adjusted Homographhy matrix
    #                                 f_square = -((H_pi[0,0] * H_pi[0,1] + H_pi[1,0] * H_pi[1,1]) / (H_pi[2,0] * H_pi[2,1]))
    #                                 beta_square = (H_pi[0,1] * H_pi[0,1] + H_pi[1,1] * H_pi[1,1] + f_square * H_pi[2,1] * H_pi[2,1]) / (H_pi[0,0] * H_pi[0,0] + H_pi[1,0] * H_pi[1,0] + f_square * H_pi[2,0] * H_pi[2,0])
                                    
    #                                 if ((beta_square < 0.5 * 0.5) or (beta_square > 2 * 2)):
    #                                     continue
    #                                 # else:
    #                                     # print('model !! H =', H)


    #                                 ###################################
    #                                 # 3.3.3 Evaluating the Model Support
    #                                 ###################################

    #                                 # transform all line segments of the model to the image

    #                                 score = 0

    #                                 court_transform = np.array(img)

    #                                 trans_court_model = np.zeros((height, width))
                                    
    #                                 for line_h in court_model_lines_h:
    #                                     start_pt_t = np.matmul(H, 
    #                                     np.array([line_h.start_pt[0], line_h.start_pt[1], 1]))
    #                                     end_pt_t = np.matmul(H, 
    #                                     np.array([line_h.end_pt[0], line_h.end_pt[1], 1]))
    #                                     start_pt_t = start_pt_t / start_pt_t[2]
    #                                     end_pt_t = end_pt_t / end_pt_t[2]


    #                                     cv2.line(trans_court_model, (int(start_pt_t[0]), int(start_pt_t[1])), (int(end_pt_t[0]), int(end_pt_t[1])), (255, 255, 255), 2)

    #                                     # Debug use
    #                                     mid_pt = (int((start_pt_t[0] + end_pt_t[0]) / 2), int((start_pt_t[1] + end_pt_t[1]) / 2))
    #                                     cv2.putText(court_transform, str(line_h.id), mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
    #                                     cv2.line(court_transform, (int(start_pt_t[0]), int(start_pt_t[1])), (int(end_pt_t[0]), int(end_pt_t[1])), (255, 255, 0), 2)

    #                                 for line_v in court_model_lines_v:
    #                                     start_pt_t = np.matmul(H, 
    #                                     np.array([line_v.start_pt[0], line_v.start_pt[1], 1]))
    #                                     end_pt_t = np.matmul(H, 
    #                                     np.array([line_v.end_pt[0], line_v.end_pt[1], 1]))
    #                                     start_pt_t = start_pt_t / start_pt_t[2]
    #                                     end_pt_t = end_pt_t / end_pt_t[2]

    #                                     cv2.line(trans_court_model, (int(start_pt_t[0]), int(start_pt_t[1])), (int(end_pt_t[0]), int(end_pt_t[1])), (255, 255, 255), 2)

    #                                     # Debug use
    #                                     mid_pt = (int((start_pt_t[0] + end_pt_t[0]) / 2), int((start_pt_t[1] + end_pt_t[1]) / 2))
    #                                     cv2.putText(court_transform, str(line_v.id), mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
    #                                     cv2.line(court_transform, (int(start_pt_t[0]), int(start_pt_t[1])), (int(end_pt_t[0]), int(end_pt_t[1])), (255, 0, 0), 2)


    #                                     trans_court_model = cv2.normalize(trans_court_model, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    #                                     # if l(x, y) -> 1
    #                                     on_court = cv2.bitwise_and(trans_court_model, trans_court_model, mask=court_line_candidate)
    #                                     # if not in l(x,y) -> -1/2
    #                                     not_on_court = cv2.bitwise_and(trans_court_model, trans_court_model, mask=court_line_candidate_invert)
    #                                     # else -> 0 (do nth)

    #                                 score += np.count_nonzero(on_court)
    #                                 score += np.count_nonzero(not_on_court) * -0.5

    #                                 if score > score_max:

    #                                     score_max = score

    #                                     saved_model = H

    #                                     cv2.imshow('court transform img', court_transform)
    #                                     cv2.imshow('trans court model', trans_court_model)
    #                                     cv2.imshow('on court', on_court)
    #                                     cv2.imshow('not on court', not_on_court)
    #                                     print('line_h_i:', line_h_i.id, ' ; line_h_k:', line_h_k.id)
    #                                     print('line_v_m:', line_v_m.id, ' ; line_v_n:', line_v_n.id)
    #                                     print('p1:', p1, ', p2:', p2, ', p3', p3, ', p4:', p4)
    #                                     print('p_cm_1:', p_cm_1, ', p_cm_2:', p_cm_2, ', p_cm_3:', p_cm_3, ', p_cm_4:', p_cm_4)
    #                                     print('H =', H)
    #                                     print('score = {}'.format(score))
    #                                     # cv2.imshow('court transform test', trans_court_model)
    #                                     cv2.waitKey(0)

    #                                 ###################################
    #                                 # Debug
    #                                 ###################################
    #                                 # court_model_img_test = np.array(court_model_img)
    #                                 # court_img_test = np.array(img)
    #                                 # line_h_i.draw_line_extended(court_img_test, (255, 0, 0))
    #                                 # line_h_k.draw_line_extended(court_img_test, (255, 0, 0))
    #                                 # line_v_m.draw_line_extended(court_img_test, (255, 0, 0))
    #                                 # line_v_n.draw_line_extended(court_img_test, (255, 0, 0))
    #                                 # cv2.circle(court_img_test, tuple(map(int, p1))[:2], 5, (0, 255, 0), -1)
    #                                 # cv2.circle(court_img_test, tuple(map(int, p2))[:2], 5, (0, 255, 85), -1)
    #                                 # cv2.circle(court_img_test, tuple(map(int, p3))[:2], 5, (0, 255, 170), -1)
    #                                 # cv2.circle(court_img_test, tuple(map(int, p4))[:2], 5, (0, 255, 255), -1)
    #                                 # cv2.circle(court_model_img_test, tuple(map(int, p_cm_1))[:2], 2, (0, 255, 0), -1)
    #                                 # cv2.circle(court_model_img_test, tuple(map(int, p_cm_2))[:2], 2, (0, 255, 85), -1)
    #                                 # cv2.circle(court_model_img_test, tuple(map(int, p_cm_3))[:2], 2, (0, 255, 170), -1)
    #                                 # cv2.circle(court_model_img_test, tuple(map(int, p_cm_4))[:2], 2, (0, 255, 255), -1)

    #                                 # cv2.imshow('court_model', court_model_img_test)
    #                                 # cv2.imshow('court_img_test', court_img_test)
    #                                 # cv2.waitKey(0)




    # showing images
    while True:
        cv2.imshow('img', img)
        cv2.imshow('court_line_cand', court_line_candidate)
        cv2.imshow('line_struct_const', line_structure_const)
        cv2.imshow('line_struct_const_and', line_structure_const_and)
        cv2.imshow('blur_canny', blur_canny)
        cv2.imshow('HoughLinesP result', img_0)
        cv2.imshow('lines extended result', img_1)
        cv2.imshow('lines extended horizontal', img_l_h)
        cv2.imshow('lines extended vertical', img_l_v)
        # cv2.imshow('line 1', img_line_1)
        # cv2.imshow('sobel line', grad)

        # for k, _img in court_line_cand_img.items():
        #     cv2.imshow('court line {}'.format(k), _img)
        
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

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
    main()
    # generate_court_model_lines()