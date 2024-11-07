import numpy as np
import cv2 as cv


def eight_point(cl, cr):
    if len(cl) < 8 or len(cr) < 8:
        print("Not enough matched corners")
        raise NotImplementedError
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
    fundamental, _, _, _ = np.linalg.lstsq(eight_matrix, minusone, rcond=None)
    fundamental = np.append(fundamental, 1)
    fundamental = np.reshape(fundamental, (3, 3)).T
    print("Fundamental Matrix")
    print(fundamental)
    return fundamental
    # print(fundamental_opencv)
    # for i in range(8):
    #     print(f"{cl[i]@fundamental@cr[i].T},{cl[i]@fundamental_opencv@cr[i].T}")


def intrinsic_matrices(side):
    if globals()[f"ret_{side}"] == True:
        refined_corners = cv.cornerSubPix(
            globals()[f"gray_{side}"],
            globals()[f"corners_{side}"],
            (11, 11),
            (-1, -1),
            criteria,
        )
        objp = np.zeros((CHESSBOARD_WIDTH * CHESSBOARD_HEIGHT, 3), np.float32)
        objp[:, :2] = np.mgrid[0:CHESSBOARD_HEIGHT, 0:CHESSBOARD_WIDTH].T.reshape(-1, 2)
        _, mtx, _, _, _ = cv.calibrateCamera(
            [objp], [refined_corners], globals()[f"gray_{side}"].shape[::-1], None, None
        )
        print(f"{side.title()} Intrinsic Matrix")
        print(mtx)
        return mtx
    else:
        print(f"Did not detect chessboard in {side} image.")
        raise NotImplementedError


CHESSBOARD_WIDTH = 7
CHESSBOARD_HEIGHT = 5

chessboard = cv.imread("StereoTest1chess.jpg")
chessboard_small = cv.resize(chessboard, (0, 0), fx=0.5, fy=0.5)

chessboard_left = chessboard_small[:, : round(chessboard_small.shape[1] / 2), :]
chessboard_right = chessboard_small[:, round(chessboard_small.shape[1] / 2) :, :]

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
gray_left = cv.cvtColor(chessboard_left, cv.COLOR_BGR2GRAY)
gray_right = cv.cvtColor(chessboard_right, cv.COLOR_BGR2GRAY)

ret_left, corners_left = cv.findChessboardCorners(
    chessboard_left, (CHESSBOARD_WIDTH, CHESSBOARD_HEIGHT), None
)
ret_right, corners_right = cv.findChessboardCorners(
    chessboard_right, (CHESSBOARD_WIDTH, CHESSBOARD_HEIGHT), None
)

left_intrinsic_matrix = intrinsic_matrices("left")
right_instrinsic_matrix = intrinsic_matrices("right")

indices = (0, 5, 10, 15, 20, 24, 29, 34)
fundamental_matrix = eight_point(corners_left[indices, :], corners_right[indices, :])
essential_matrix = (
    left_intrinsic_matrix.T @ fundamental_matrix @ right_instrinsic_matrix
)

print("Essential Matrix")
print(essential_matrix)

# essential_opencv, _ = cv.findEssentialMat(
#     corners_left[indices, :],
#     corners_right[indices, :],
#     cameraMatrix=(left_intrinsic_matrix + right_instrinsic_matrix) / 2,
#     method=cv.RANSAC,
# )
# print(essential_opencv)
ret, r, t, _ = cv.recoverPose(
    essential_matrix,
    corners_left[indices, :],
    corners_right[indices, :],
    cameraMatrix=(left_intrinsic_matrix + right_instrinsic_matrix) / 2,
)
print("Rotation")
print(r)
print("Translation")
print(t)
