import numpy as np

from line import Line

class TennisCourtModel:

    # times two to prevent drawing error in opencv
    y = np.array([0, 18, 60, 78]) * 2
    x = np.array([0, 4.5, 18, 31.5, 36]) * 2

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

    