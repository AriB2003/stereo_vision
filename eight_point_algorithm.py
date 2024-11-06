import numpy as np
import cv2 as cv


def eight_point(cl, cr):
    if len(cl) < 8 or len(cr) < 8:
        print("Not enough matched corners")
        return
    cl = cl[:8]
    cr = cr[:8]
    fundamental_opencv, _ = cv.findFundamentalMat(cl, cr, method=cv.FM_8POINT)
    one = np.ones((len(cl), 1, 3))
    one[:, :, :2] = cl
    cl = one
    one = np.ones((len(cr), 1, 3))
    one[:, :, :2] = cr
    cr = one
    zero = np.zeros((8, 8))
    for i in range(8):
        row = cl[i].T @ cr[i]
        row = np.reshape(row, (1, -1))
        zero[i, :] = row[:, :8]
    eight_matrix = zero
    minusone = -1 * np.ones((8, 1))
    fundemental, _, _, _ = np.linalg.lstsq(eight_matrix, minusone, rcond=None)
    fundemental = np.append(fundemental, 1)
    fundemental = np.reshape(fundemental, (3, 3)).T
    print(fundemental)
    print(fundamental_opencv)
    for i in range(8):
        print(f"{cl[i]@fundemental@cr[i].T},{cl[i]@fundamental_opencv@cr[i].T}")


CHESSBOARD_WIDTH = 7
CHESSBOARD_HEIGHT = 5

chessboard = cv.imread("StereoTest1chess.jpg")
chessboard_small = cv.resize(chessboard, (0, 0), fx=0.5, fy=0.5)

chessboard_left = chessboard_small[:, : round(chessboard_small.shape[1] / 2), :]
chessboard_right = chessboard_small[:, round(chessboard_small.shape[1] / 2) :, :]

ret_left, corners_left = cv.findChessboardCorners(
    chessboard_left, (CHESSBOARD_WIDTH, CHESSBOARD_HEIGHT), None
)
ret_right, corners_right = cv.findChessboardCorners(
    chessboard_right, (CHESSBOARD_WIDTH, CHESSBOARD_HEIGHT), None
)

print(f"{ret_left},{ret_right}")
# print(f"{corners_left}\n{corners_right}")

indices = (0, 5, 10, 15, 20, 24, 29, 34)
eight_point(corners_left[indices, :], corners_right[indices, :])
