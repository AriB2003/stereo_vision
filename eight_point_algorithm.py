import numpy as np
import cv2 as cv
import glob

CHESSBOARD_WIDTH = 7
CHESSBOARD_HEIGHT = 5

CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


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
    print("Custom Fundamental Matrix")
    print(fundamental)
    print("OpenCV Fundamental Matrix")
    print(fundamental_opencv)
    # for i in range(8):
    #     print(f"{cl[i]@fundamental@cr[i].T},{cl[i]@fundamental_opencv@cr[i].T}")
    return fundamental


def intrinsic_matrices(ret, gray, corners, tit):
    if ret == True:
        refined_corners = cv.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            CRITERIA,
        )
        objp = np.zeros((CHESSBOARD_WIDTH * CHESSBOARD_HEIGHT, 3), np.float32)
        objp[:, :2] = np.mgrid[0:CHESSBOARD_HEIGHT, 0:CHESSBOARD_WIDTH].T.reshape(-1, 2)
        _, mtx, _, _, _ = cv.calibrateCamera(
            [objp], [refined_corners], gray.shape[::-1], None, None
        )
        print(f"{tit.title()} Intrinsic Matrix")
        print(mtx)
        return mtx
    else:
        print(f"Did not detect chessboard in {tit} image.")
        raise NotImplementedError


def run_camera_calibration():
    left_intrinsic_matrix = np.zeros((3, 3))
    right_intrinsic_matrix = np.zeros((3, 3))
    corners_left = None
    corners_right = None
    left_counter = 0
    right_counter = 0

    for filename in glob.glob("zed_calibration/*"):
        print(filename)
        chessboard = cv.imread(filename)
        chessboard_small = cv.resize(chessboard, (0, 0), fx=0.5, fy=0.5)

        if not "l" in filename.split("\\")[-1]:
            # Right or Both
            chessboard_right = chessboard_small[
                :, round(chessboard_small.shape[1] / 2) :, :
            ]
            gray_right = cv.cvtColor(chessboard_right, cv.COLOR_BGR2GRAY)
            ret_right, corners_right = cv.findChessboardCorners(
                chessboard_right, (CHESSBOARD_WIDTH, CHESSBOARD_HEIGHT), None
            )
            right_intrinsic_matrix += intrinsic_matrices(
                ret_right, gray_right, corners_right, "right"
            )
            right_counter += 1
        if not "r" in filename.split("\\")[-1]:
            # Right or Both
            chessboard_left = chessboard_small[
                :, : round(chessboard_small.shape[1] / 2), :
            ]
            gray_left = cv.cvtColor(chessboard_left, cv.COLOR_BGR2GRAY)
            ret_left, corners_left = cv.findChessboardCorners(
                chessboard_left, (CHESSBOARD_WIDTH, CHESSBOARD_HEIGHT), None
            )
            left_intrinsic_matrix += intrinsic_matrices(
                ret_left, gray_left, corners_left, "left"
            )
            left_counter += 1

    left_intrinsic_matrix /= left_counter
    right_intrinsic_matrix /= right_counter

    print("Final Left Intrinsic Matrix")
    print(left_intrinsic_matrix)
    print("Final Right Intrinsic Matrix")
    print(right_intrinsic_matrix)

    indices = (0, 5, 10, 15, 20, 24, 29, 34)

    # corners_left[:, 0, 0] = (
    #     corners_left[:, 0, 0] - left_intrinsic_matrix[0, 2]
    # ) / left_intrinsic_matrix[0, 0]
    # corners_left[:, 0, 1] = (
    #     corners_left[:, 0, 1] - left_intrinsic_matrix[1, 2]
    # ) / left_intrinsic_matrix[1, 1]
    # corners_right[:, 0, 0] = (
    #     corners_right[:, 0, 0] - right_intrinsic_matrix[0, 2]
    # ) / right_intrinsic_matrix[0, 0]
    # corners_right[:, 0, 1] = (
    #     corners_right[:, 0, 1] - right_intrinsic_matrix[1, 2]
    # ) / right_intrinsic_matrix[1, 1]
    # print(corners_left)

    fundamental_matrix = eight_point(
        corners_left[indices, :], corners_right[indices, :]
    )
    essential_matrix = (
        left_intrinsic_matrix.T @ fundamental_matrix @ left_intrinsic_matrix
    )

    corners_left = cv.undistortPoints(corners_left, left_intrinsic_matrix, None)
    corners_right = cv.undistortPoints(corners_right, right_intrinsic_matrix, None)
    # print(corners_left)
    # print(corners_right)

    essential_opencv, _ = cv.findEssentialMat(
        corners_left[indices, :],
        corners_right[indices, :],
        cameraMatrix=left_intrinsic_matrix,
    )

    print("Essential Matrix")
    print(essential_matrix)
    print("OpenCV Essential Matrix")
    print(essential_opencv)
    ret, r, t, _ = cv.recoverPose(
        essential_matrix,
        corners_left[indices, :],
        corners_right[indices, :],
        cameraMatrix=(left_intrinsic_matrix + right_intrinsic_matrix) / 2,
    )
    print("Rotation")
    print(r)
    print("Translation")
    print(t)
    return (
        left_intrinsic_matrix,
        right_intrinsic_matrix,
        fundamental_matrix,
        essential_matrix,
    )
