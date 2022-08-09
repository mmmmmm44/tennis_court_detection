import math
import cv2
import numpy as np

# const parameter config
from config import DEGREE_THRESH, DIST_LINES_THRESH

# Line object that stores its equation and parameterized parameters

class Line:
    def __init__(self, id, A, B, C):
        self.id = id
        # define a line by Ax + By + C = 0
        self.A = A
        self.B = B
        self.C = C

        # pre-cal
        self.COS_DEGREE_THRESH = math.cos(DEGREE_THRESH * math.pi / 180)

    @classmethod
    def from_two_point(cls, id, p1, p2):
        '''
        Define a line with its two point form
        '''
        # for court model
        A, B, C = Line.eqt_of_two_point(p1, p2)
        obj = cls(id, A, B, C)
        obj.start_pt = p1
        obj.end_pt = p2
        return obj

    @classmethod
    def from_point_slope(cls, id, p1, m):
        A, B, C = m, -1, -m * p1[0] + p1[1]

        return cls(id, A, B, C)

    # def __init__(self, A, B, C):
    #     self.A = A
    #     self.B = B
    #     self.C = C

    def line_parameterization(self):
        '''
        Parameterize a line to format (n_x, n_y, -d)
        where n_x, n_y is unit vector of the normal of the line
        and d = perpendicular distance to the origin
        '''
        slope_original = -self.A / self.B
        self.m_x, self.m_y = Line.slope_to_unit_vector(slope_original)

        slope_normal = self.B / self.A
        self.n_x, self.n_y = Line.slope_to_unit_vector(slope_normal)

        # check whether the vectors are normal to each other
        # print('Line: {} dot product btw original and normal: {}'.format(self.id, np.vdot(np.array([m_x,m_y]), np.array([self.n_x, self.n_y]))))

        self.d = Line.dist_btw_line_point(self.A, self.B, self.C, (0, 0))

        self.parameterized = [self.n_x, self.n_y, -self.d]

    def __eq__(self, other):
        return self.is_duplicate(other)

    # check whether two lines are duplicates
    def is_duplicate(self, other):
        if ((np.dot(self.parameterized[:2], other.parameterized[:2]) > self.COS_DEGREE_THRESH) and (abs(self.parameterized[2] - other.parameterized[2]) < DIST_LINES_THRESH)):
            return True
        else:
            return False


    ####################
    # Visualization
    ####################

    def draw_line(self, img, color):
        '''
        Draw the line defined by the start and end point
        '''
        mid_pt = (int((self.start_pt[0] + self.end_pt[0]) / 2.0), int((self.start_pt[1] + self.end_pt[1]) / 2.0))
        cv2.putText(img, str(self.id), mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.line(img, (int(self.start_pt[0]), int(self.start_pt[1])), (int(self.end_pt[0]), int(self.end_pt[1])), color, 2)

    def draw_normal(self, img, color, length=50, thickness=2):
        '''
        Draw the parameterized normal at the mid-pt of the line
        with self-defined length
        '''
        mid_pt = (int((self.start_pt[0] + self.end_pt[0]) / 2.0), int((self.start_pt[1] + self.end_pt[1]) / 2.0))
        end_pt = (int(mid_pt[0] + self.n_x * length), int(mid_pt[1] + self.n_y * length))
        cv2.line(img, mid_pt, end_pt, color, thickness)


    def __select_edge_pts(self, edge_pts, width, height):
        candidate = []
        for pt in edge_pts:
            if (((0 <= pt[0]) and (pt[0] <= width)) and ((0 <= pt[1]) and (pt[1] <= height))):
                candidate.append(pt)
        return candidate

    def draw_line_extended(self, img, color, thickness=2):
        '''
        Draw a line that is extended to the edges of the image
        '''
        height, width, _ = img.shape
        edge_pts = [[0, -self.C/self.B], [-self.C/self.A, 0], [width, -self.A / self.B * width - self.C/self.B], [-self.B/self.A * height -self.C/self.A, height]]   # from Ax + By + C = 0
        edge_pts_n = self.__select_edge_pts(edge_pts, width, height)
        self.start_pt, self.end_pt = min(edge_pts_n, key=lambda x: x[0]), max(edge_pts_n, key=lambda x: x[0])

        # print('line id: {}, eqt = {}x+{}y+{}=0, slope: {:.4f}, orientation: {:.4f}'.format(
        #     self.id, self.A, self.B, self.C, -self.A/self.B, 
        #     math.degrees(math.atan2(end_pt[1] - start_pt[1], end_pt[0] - start_pt[0]))
        #     ))

        mid_pt = (int((self.start_pt[0] + self.end_pt[0]) / 2.0), int((self.start_pt[1] + self.end_pt[1]) / 2.0))

        # draw
        cv2.putText(img, str(self.id), mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.line(img, (int(self.start_pt[0]), int(self.start_pt[1])), (int(self.end_pt[0]), int(self.end_pt[1])), color, thickness)

    def __str__(self) -> str:
        s = 'ID: {}; Eqt: {}x + {}y + {} = 0'.format(self.id, self.A, self.B, self.C)
        return s

    ####################
    # Helper functions
    ####################

    def eqt_of_two_point(p1, p2):
        '''
        Return general form coefficients of the line passing through p1 and p2
        the line is Ax + By + C = 0
        Ref: https://stackoverflow.com/questions/13242738/how-can-i-find-the-general-form-equation-of-a-line-from-two-points
        '''
        A = p2[1] - p1[1]
        B = -(p2[0] - p1[0])
        C = -(p1[0] * p2[1] - p2[0] * p1[1])

        return A, B, C

    def slope_to_unit_vector(m):
        '''
        Solving the equation with two unknowns: x, y
        (y - 0)/(x - 0) = m
        x^2 + y^2 = 1
        '''

        # x = math.sqrt(1 / (1 + m * m))
        # y = math.sqrt(1 / (1 + 1 / (m * m)))

        # return x, y

        vec = np.array([1, m])
        normalized = vec / np.linalg.norm(vec)

        # check if m is inf or -inf
        if m == float('inf') or m == float('-inf'):
            normalized = np.array([0, 1])

        return normalized[0], normalized[1]        

    def dist_btw_line_point(A, B, C, p1):
        '''
        The distance between a line Ax + By + C = 0 and a point (x, y)
        Ref: https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        '''
        return abs(A * p1[0] + B * p1[1] + C) / math.sqrt(A * A + B * B)


    def solve_intersection(line_A, line_B):
        '''
        Solve intersection of two lines.
        '''
        # handle determinant = 0

        a = np.array([[line_A.A, line_A.B], [line_B.A, line_B.B]])
        b = np.array([-line_A.C, -line_B.C])
        return np.linalg.solve(a, b)    


