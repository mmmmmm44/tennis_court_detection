import cv2
import numpy as np

from tennis_court_model import TennisCourtModel

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