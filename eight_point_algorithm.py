""" This module runs camera calibration on the test images and computer the essential, fundamental, and intrinsic matrices. """

import numpy as np
import cv2 as cv
import glob

# Set chessboard size global constants
CHESSBOARD_WIDTH = 7
CHESSBOARD_HEIGHT = 5

# Set corner refinement criteria
CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def eight_point(cl, cr, debug=True):
    """
    Run the eight-point algorithm on two sets of points.

    Args:
        cl: points in the left image, 2d np mx2 [[x,y]...]
        cr: points in the right image, 2d np mx2 [[x,y]...]
        debug: enable debug printing, bool

    Returns:
        fundamental: computed fundamental matrix, np 2d 3x3
    """
    # Check input validity
    if len(cl) < 8 or len(cr) < 8:
        print("Not enough matched corners")
        raise NotImplementedError
    # Cut the input to only eight points
    cl = cl[:8]
    cr = cr[:8]
    # Run built-in OpenCV eight-point algorithm for verification
    fundamental_opencv, _ = cv.findFundamentalMat(cl, cr, method=cv.FM_8POINT)

    # Add a column of ones to the points to homogenize them
    one = np.ones((len(cl), 1, 3))
    one[:, :, :2] = cl
    cl = one
    one = np.ones((len(cr), 1, 3))
    one[:, :, :2] = cr
    cr = one

    # Calculate the 8x8 matrix for the eight-point algorithm (A in Ax=b)
    zero = np.zeros((8, 8))
    for i in range(8):
        # Set each row as the unrolled 3x3 product of left points and right points
        row = cl[i].T @ cr[i]
        row = np.reshape(row, (1, -1))
        # Crop the homogenous column off of the matrix
        zero[i, :] = row[:, :8]
    eight_matrix = zero

    # Set the resultant vector to an 8x1 of -1s (b in Ax=b)
    minusone = -1 * np.ones((8, 1))

    # Calculate the least squares approximation to find x in Ax=b
    fundamental, _, _, _ = np.linalg.lstsq(eight_matrix, minusone, rcond=None)
    # Append a 1 to the end to force the scale solution to a fixed solution
    fundamental = np.append(fundamental, 1)
    # Unroll x into a 3x3 matrix, which is the fundamental matrix
    fundamental = np.reshape(fundamental, (3, 3)).T

    if debug:
        print("Custom Fundamental Matrix")
        print(fundamental)
        print("OpenCV Fundamental Matrix")
        print(fundamental_opencv)

    return fundamental


def intrinsic_matrices(ret, gray, corners, tit, debug=True):
    """
    Calculate the intrinsic matrix from an image of a calibration chessboard.

    Args:
        ret: chessboard was detected, bool
        gray: grayscale image, np 3d mxnx1
        corners: points of the chessboard, 2d np mx2 [[x,y]...]
        tit: title of the printout, string
        debug: enable debug printing, bool

    Returns:
        mtx: computed intrinsic matrix, np 2d 3x3
    """
    if ret == True:
        # Iteratively refine the corners to decimal values
        refined_corners = cv.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            CRITERIA,
        )

        # Create objects
        objp = np.zeros((CHESSBOARD_WIDTH * CHESSBOARD_HEIGHT, 3), np.float32)
        objp[:, :2] = np.mgrid[0:CHESSBOARD_HEIGHT, 0:CHESSBOARD_WIDTH].T.reshape(-1, 2)

        # Run camera calibration and extract intrinsic matrix
        _, mtx, _, _, _ = cv.calibrateCamera(
            [objp], [refined_corners], gray.shape[::-1], None, None
        )

        if debug:
            print(f"{tit.title()} Intrinsic Matrix")
            print(mtx)
        return mtx
    else:
        print(f"Did not detect chessboard in {tit} image.")
        raise NotImplementedError


def run_camera_calibration(debug=True):
    """
    Run the full camera calibration procedure:

    Args:
        debug: enable debug printing, bool

    Returns:
        left_intrinsic_matrix: intrinsic matrix K for left camera, np 2d 3x3
        right_intrinsic_matrix: intrinsic matrix K for right camera, np 2d 3x3
        fundamental_matrix: fundamental matrix F for stereo cameras, np 2d 3x3
        essential_matrix: essential matrix E for stereo cameras, np 2d 3x3
    """
    # Initialize values
    left_intrinsic_matrix = np.zeros((3, 3))
    right_intrinsic_matrix = np.zeros((3, 3))
    corners_left = None
    corners_right = None
    left_counter = 0
    right_counter = 0

    # Loop through calibration images
    for filename in glob.glob("zed_calibration/*"):
        if debug:
            print(filename)

        # Load image and scale down for speed
        chessboard = cv.imread(filename)
        chessboard_small = cv.resize(chessboard, (0, 0), fx=0.5, fy=0.5)

        if not "l" in filename.split("\\")[-1]:
            # If right calibration image or joint calibration
            # Crop to right side of image
            chessboard_right = chessboard_small[
                :, round(chessboard_small.shape[1] / 2) :, :
            ]
            # Convert to grayscale
            gray_right = cv.cvtColor(chessboard_right, cv.COLOR_BGR2GRAY)
            # Find the chessboard corners
            ret_right, corners_right = cv.findChessboardCorners(
                chessboard_right, (CHESSBOARD_WIDTH, CHESSBOARD_HEIGHT), None
            )
            # Calculate the intrinsic matrix
            right_intrinsic_matrix += intrinsic_matrices(
                ret_right, gray_right, corners_right, "right", debug=debug
            )
            # Count the right intrinsic matrices calibrated against
            right_counter += 1
        if not "r" in filename.split("\\")[-1]:
            # If left calibration image or joint calibration
            # Crop to left side of image
            chessboard_left = chessboard_small[
                :, : round(chessboard_small.shape[1] / 2), :
            ]
            # Convert to grayscale
            gray_left = cv.cvtColor(chessboard_left, cv.COLOR_BGR2GRAY)
            # Find the chessboard corners
            ret_left, corners_left = cv.findChessboardCorners(
                chessboard_left, (CHESSBOARD_WIDTH, CHESSBOARD_HEIGHT), None
            )
            # Calculate the intrinsic matrix
            left_intrinsic_matrix += intrinsic_matrices(
                ret_left, gray_left, corners_left, "left", debug=debug
            )
            # Count the left intrinsic matrices calibrated against
            left_counter += 1

    # Average the intrinsic matrices
    left_intrinsic_matrix /= left_counter
    right_intrinsic_matrix /= right_counter

    if debug:
        print("Final Left Intrinsic Matrix")
        print(left_intrinsic_matrix)
        print("Final Right Intrinsic Matrix")
        print(right_intrinsic_matrix)

    # Select well-spaced indices for the eight-point algorithm
    indices = (0, 5, 10, 15, 20, 24, 29, 34)

    # Perform eight-point calibration
    fundamental_matrix = eight_point(
        corners_left[indices, :], corners_right[indices, :], debug=debug
    )

    # Calculate the essential matrix from E=K1'*F*K2
    essential_matrix = (
        left_intrinsic_matrix.T @ fundamental_matrix @ right_intrinsic_matrix
    )

    # Undistort the points
    corners_left = cv.undistortPoints(corners_left, left_intrinsic_matrix, None)
    corners_right = cv.undistortPoints(corners_right, right_intrinsic_matrix, None)

    # Run the built-in OpenCV essential matrix to compare
    essential_opencv, _ = cv.findEssentialMat(
        corners_left[indices, :],
        corners_right[indices, :],
        cameraMatrix=left_intrinsic_matrix,
    )

    if debug:
        print("Essential Matrix")
        print(essential_matrix)
        print("OpenCV Essential Matrix")
        print(essential_opencv)

    # Recover the pose for debugging purposes
    ret, r, t, _ = cv.recoverPose(
        essential_matrix,
        corners_left[indices, :],
        corners_right[indices, :],
        cameraMatrix=(left_intrinsic_matrix + right_intrinsic_matrix) / 2,
    )

    if debug:
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


if __name__ == "__main__":
    run_camera_calibration(debug=True)
