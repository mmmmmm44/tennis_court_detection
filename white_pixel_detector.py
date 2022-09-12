import cv2
import numpy as np

import itertools
import multiprocessing as mp

from config import TAU, THRESHOLD_L, THRESHOLD_D, BLOCK_SIZE, APERATURE_SIZE

# Section 3.1 White Pixel Detection

class WhitePixelDetector:

    def __init__(self) -> None:
        self.court_line_candidate = None
        self.line_structure_const = None
        self.line_structure_const_and = None

    def execute(self, img):
        '''
        Section 3.1 White Pixel detection

        img: BGR image
        return: image with detected white line pixel
        '''

        height, width = img.shape[:2]

        img_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

        # prevent overflow error
        img_y_int32 = img_ycbcr[:, :, 0].astype(np.int32)

        tau = TAU
        threshold_l = THRESHOLD_L
        threshold_d = THRESHOLD_D

        # about 50% faster (3.3x sec -> 1.7x sec) than single process version
        self.court_line_candidate = WhitePixelDetector._get_court_line_candidate_mp(img_y_int32, height, width, tau, threshold_l, threshold_d)

        # exclude pixels that are in textured regions
        
        # previously implemented CourtLinePixelDetector::computeStructureTensorElements from https://github.com/gchlebus/tennis-court-detection
        # result: not as good as mine

        img_y = img_ycbcr[:, :, 0]

        block_size = BLOCK_SIZE             # affect the most in edge/ surface detection
        aperture_size = APERATURE_SIZE
        structure_matrix = cv2.cornerEigenValsAndVecs(
            img_y,
            block_size,
            aperture_size
        )

        # single process is faster than multi-process by around 0.5 sec (2.6s vs 3.1s)
        self.line_structure_const = np.zeros((height, width))
        for x in range(width):
            for y in range(height):
                self.line_structure_const[y, x] = WhitePixelDetector._line_struct_const_formula(structure_matrix, y, x)

        # normalize to [0, 255] range with correct datatype (np.uint8)
        self.court_line_candidate = cv2.normalize(self.court_line_candidate, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.line_structure_const = cv2.normalize(self.line_structure_const, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        self.line_structure_const_and = cv2.bitwise_and(self.court_line_candidate, self.court_line_candidate, mask=self.line_structure_const)

        return self.line_structure_const_and


    def _get_court_line_candidate_mp(int_y_int32, height, width, tau, threshold_l, threshold_d):
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
        results = pool.starmap(WhitePixelDetector._court_line_formula, args)

        # unpack the results into the matrix
        for i_tuple, result in zip([(row, col) for (row, col) in list(itertools.product(range(height), range(width)))], results):
            # unpack
            r, c = i_tuple

            # set it in the matrix
            court_line_candidate[r, c] = result

        return court_line_candidate


    def _court_line_formula(img_y, y, x, tau, threshold_l, threshold_d, height, width):
        '''
        Forumla for determining whether a pixel is court line candidate
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


    def _line_struct_const_formula(structure_matrix, y, x):
        '''
        Formula for determining whether a pixel belongs to region or a line
        by comparing eigenvalues of structure matrix. (line structure constraint)
        in 3.1 White pixel detection
        '''
        lambda_max, lambda_min = max(structure_matrix[y, x, 0], structure_matrix[y, x, 1]), min(structure_matrix[y, x, 0], structure_matrix[y, x, 1])

        if (lambda_max > 4 * lambda_min): return 1
        else: return 0