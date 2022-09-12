import math
from tennis_court_model import TennisCourtModel

import cv2
import numpy as np

from line import Line
from config import CLS_ANGLE_THRESH

class ModelFitting:
    def __init__(self) -> None:
        
        # get court model line parameters
        self.court_model_h, self.court_model_v, self.court_model_lines_h, self.court_model_lines_v = TennisCourtModel.y, TennisCourtModel.x, TennisCourtModel.court_model_lines_h, TennisCourtModel.court_model_lines_v


    def execute(self, img, lines_extended, line_structure_const_and):
        '''
        Section 3.3: Model Fitting

        Involve three steps: Finding line correspondences, Fast Calibration Parameter Rejection Test, Evaluating the Model Support

        Return: the best Homography matrix, and its score
        '''

        self.img = img
        self.lines_extended = lines_extended
        self.line_structure_const_and = line_structure_const_and

        # assume the processing images remain the same shape as the original image
        self.height, self.width = self.line_structure_const_and.shape[:2]

        ###################################
        # 3.3.1 Finding Line Correspondences
        ###################################
        self.lines_horizontal, self.lines_vertical = self._find_line_correspondences()

        print('Num of lines_horizontal:', len(self.lines_horizontal))
        print('Num of lines_vertical:', len(self.lines_vertical))


        ###################################
        # 3.3.2 & 3.3.3 Fast Calibration Parameter Rejection Test & Evaluating the Model Support
        ###################################
        best_model, score_max = self._find_best_line_assignment()

        return best_model, score_max


    # Section 3.1.1
    def _find_line_correspondences(self):

        # create two sets, one is horizontal, one is vertical
        lines_horizontal, lines_vertical = [], []
        degree_h = CLS_ANGLE_THRESH
        degree_h_in_rad = degree_h * math.pi / 180

        for line in self.lines_extended.values():
            result_pyatan2 = math.atan2(line.m_y, line.m_x)
            if abs(result_pyatan2) < degree_h_in_rad:
                lines_horizontal.append(line)
            else:
                lines_vertical.append(line)


        # sort the lines

        # define the sorting function
        def sort_dist_key(line_para, p1):
            return np.dot(line_para, p1)

        lines_horizontal.sort(
            key = lambda x: sort_dist_key(x.get_parameterized(), np.array([int(self.width / 2.0), 0, 1])), reverse=True
        )
        lines_vertical.sort(
            key = lambda x: sort_dist_key(x.get_parameterized(), np.array([0, int(self.height / 2.0), 1])), reverse=True
        )


        # Moved the check duplicates to 
        # after classifying the lines into two groups, horizontal and vertical
        last = ModelFitting.unique(lines_horizontal)
        lines_horizontal = lines_horizontal[:last]
        
        last = ModelFitting.unique(lines_vertical)
        lines_vertical = lines_vertical[:last]

        return lines_horizontal, lines_vertical
    

    # section 3.2 and 3.3
    def _find_best_line_assignment(self):

        # init parameters and variables required
        best_model = None

        compensate_matrix = np.array(
            [[1, 0, -self.width / 2.0],
            [0, 1, -self.height / 2.0],
            [0, 0, 1]]
        )

        score_max = float('-inf')

        line_structure_const_and_invert = np.invert(self.line_structure_const_and)


        for i in range(len(self.lines_horizontal)):
            for k in range(i + 1, len(self.lines_horizontal)):
                for m in range(len(self.lines_vertical)):
                    for n in range(m + 1, len(self.lines_vertical)):
                        line_h_i = self.lines_horizontal[i]
                        line_h_k = self.lines_horizontal[k]
                        line_v_m = self.lines_vertical[m]
                        line_v_n = self.lines_vertical[n]

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

                        for ii in range(len(self.court_model_h)):
                            for kk in range(ii + 1, len(self.court_model_h)):
                                for mm in range(len(self.court_model_v)):
                                    for nn in range(mm + 1, len(self.court_model_v)):
                                
                                        line_cm_v1 = self.court_model_v[mm]
                                        line_cm_v2 = self.court_model_v[nn]
                                        line_cm_h1 = self.court_model_h[ii]
                                        line_cm_h2 = self.court_model_h[kk]

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
                                        trans_court_model = np.zeros((self.height, self.width))

                                        # Debug only
                                        # court_transform = np.array(self.img)

                                        
                                        for line_h in self.court_model_lines_h:
                                            start_pt_t = np.matmul(H, np.array([line_h.start_pt[0], line_h.start_pt[1], 1]))
                                            end_pt_t = np.matmul(H, np.array([line_h.end_pt[0], line_h.end_pt[1], 1]))
                                            start_pt_t = start_pt_t / start_pt_t[2]
                                            end_pt_t = end_pt_t / end_pt_t[2]

                                            cv2.line(trans_court_model, (int(start_pt_t[0]), int(start_pt_t[1])), (int(end_pt_t[0]), int(end_pt_t[1])), (255, 255, 255), 2)

                                            # Debug use
                                            # mid_pt = (int((start_pt_t[0] + end_pt_t[0]) / 2), int((start_pt_t[1] + end_pt_t[1]) / 2))
                                            # cv2.putText(court_transform, str(line_h.id), mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
                                            # cv2.line(court_transform, (int(start_pt_t[0]), int(start_pt_t[1])), (int(end_pt_t[0]), int(end_pt_t[1])), (255, 255, 0), 2)

                                        for line_v in self.court_model_lines_v:
                                            start_pt_t = np.matmul(H, np.array([line_v.start_pt[0], line_v.start_pt[1], 1]))
                                            end_pt_t = np.matmul(H, np.array([line_v.end_pt[0], line_v.end_pt[1], 1]))
                                            start_pt_t = start_pt_t / start_pt_t[2]
                                            end_pt_t = end_pt_t / end_pt_t[2]

                                            cv2.line(trans_court_model, (int(start_pt_t[0]), int(start_pt_t[1])), (int(end_pt_t[0]), int(end_pt_t[1])), (255, 255, 255), 2)

                                            # Debug use
                                            # mid_pt = (int((start_pt_t[0] + end_pt_t[0]) / 2), int((start_pt_t[1] + end_pt_t[1]) / 2))
                                            # cv2.putText(court_transform, str(line_v.id), mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
                                            # cv2.line(court_transform, (int(start_pt_t[0]), int(start_pt_t[1])), (int(end_pt_t[0]), int(end_pt_t[1])), (255, 0, 0), 2)


                                        trans_court_model = cv2.normalize(trans_court_model, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                                        # if l(x, y) -> 1
                                        on_court = cv2.bitwise_and(trans_court_model, trans_court_model, mask=self.line_structure_const_and)
                                        # if not in l(x,y) -> -1/2
                                        not_on_court = cv2.bitwise_and(trans_court_model, trans_court_model, mask=line_structure_const_and_invert)
                                        # else -> 0 (do nth)

                                        score += np.count_nonzero(on_court)
                                        score += np.count_nonzero(not_on_court) * -0.5

                                        if score > score_max:
                                            score_max = score
                                            best_model = H

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

        return best_model, score_max


    
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

    