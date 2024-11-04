import numpy as np
import cv2 as cv

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
width = 7
height = 5
objp = np.zeros((width * height, 3), np.float32)
objp[:, :2] = np.mgrid[0:height, 0:width].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# img = cv.imread("chessboard.jpeg")
img = cv.imread("StereoTest1chess.jpg")
img = cv.resize(img, (0, 0), fx=0.5, fy=0.5)
img = img[:, : round(img.shape[1] / 2), :]
# img = img[:, round(img.shape[1] / 2) :, :]
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Find the chess board corners
ret, corners = cv.findChessboardCorners(gray, (width, height), None)

# If found, add object points, image points (after refining them)
if ret == True:
    objpoints.append(objp)

    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners2)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    print(mtx)

    # Draw and display the corners
    cv.drawChessboardCorners(img, (width, height), corners2, bool(ret))
    cv.imshow("img", img)
    cv.waitKey(-5)

cv.destroyAllWindows()
