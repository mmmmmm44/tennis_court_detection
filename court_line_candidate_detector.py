import numpy as np

from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage import img_as_ubyte, img_as_float

from scipy import stats

from line import Line
from config import HOUGH_THRESHOLD, THRESHOLD_R

# Section 3.2 Court Line Candidate Detector

class CourtLineCandidateDetector:
    def __init__(self, img, line_structure_const_and) -> None:
        # init parameters to be accessed
        self.img = np.array(img)
        self.line_structure_const_and = line_structure_const_and
        self.height, self.width = img.shape[:2]

        self.blur_canny = None


    def execute(self):
        '''
        Section 3.2: Court line candidate detector

        line_structure_const_and: detected white pixels with line structure constraint.
        return: a dictionary of lines
        '''

        self.lines = {}

        temp = img_as_float(self.line_structure_const_and)
        self.blur_canny = canny(temp, sigma=3)

        # hough line transform from skimage
        tested_angles = np.linspace(-np.pi, np.pi, 360, endpoint=False)
        h, theta, d = hough_line(self.blur_canny, theta=tested_angles)

        thresh_hough = HOUGH_THRESHOLD * np.amax(h)

        for id, (_, angle, dist) in enumerate(zip(*hough_line_peaks(h, theta, d, threshold=thresh_hough))):
            (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
            slope = np.tan(angle + np.pi/2)

            # draw a line on opencv
            l = Line.from_point_slope(id, (x0, y0), slope)
            self.lines[id] = l
            # l.draw_line_extended(img_1, (255, 0, 0))

        # convert the image back to numpy array compatible for opencv
        self.blur_canny = img_as_ubyte(self.blur_canny)


        # parameterized every lines by its normal (normalized, start coordinate is (0, 0))
        for line in self.lines.values():
            line.line_parameterization()

        # the set L of court line candidate pixels for each line
        self.court_line_candidate_pixels = {}
        dist_thresh = THRESHOLD_R

        for y in range(self.height):
            for x in range(self.width):
                # check for zero pixels
                if self.line_structure_const_and[y, x] < 127:
                    continue

                for line in self.lines.values():
                    p = np.array([x, y, 1])
                    q = np.array(line.parameterized)
                    if abs(np.dot(q, p)) < dist_thresh:

                        if line.id not in self.court_line_candidate_pixels:
                            self.court_line_candidate_pixels[line.id] = [[], []]

                        # court_line_candidate_pixels[line.id].append([x, y])
                        self.court_line_candidate_pixels[line.id][0].append(x)
                        self.court_line_candidate_pixels[line.id][1].append(y)


        # Apply LMedS estimator. Yet I cannot find any python implementation.
        # A similar one with same % breakdown point will be "Robust Regression Using Repeated Medians".
        # scipy.stats.siegelslopes implements the paper.

        self.refinded_lines = {}

        for key, points in self.court_line_candidate_pixels.items():
            x = points[0]
            y = points[1]
            m, c = stats.siegelslopes(y, x)

            regressed = Line.from_point_slope(id=key, p1=(0, c), m=m)
            
            # line paramaterization for the new line
            regressed.line_parameterization()

            self.refinded_lines[key] = regressed

        return self.refinded_lines


    ########################################
    # Visualization Methods (intermediate processes)
    ########################################

    def court_line_cand_pixels_imgs(self):
        '''
        Return a set of images representing the court line candidate pixels
        that are close to each line
        '''
        court_line_cand_img = {}

        for k in self.court_line_candidate_pixels.keys():
            img_line = np.zeros((self.height, self.width))
            lists = self.court_line_candidate_pixels[k]
            for p in zip(lists[0], lists[1]):
                img_line[p[1], p[0]] = 1

            court_line_cand_img[k] = img_line


        return court_line_cand_img


    def regression_before_and_after_imgs(self):
        '''
        Return a set of images showing the before and after regression is applied
        of each lines
        '''
        regression_ba = {}

        for key in self.court_line_candidate_pixels.keys():
            img_siegelslope = np.array(self.img)
            self.lines_extended[key].draw_line_extended(img_siegelslope, (255, 0, 0))
            self.refinded_lines[key].draw_line_extended(img_siegelslope, (0, 0, 255))
            # draw the normal
            self.lines_extended[key].draw_normal(img_siegelslope, (255, 127, 127), length=80, thickness=2)

            regression_ba[key] = img_siegelslope

        return regression_ba
