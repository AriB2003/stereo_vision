from PIL import Image
import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from eight_point_algorithm import run_camera_calibration


def decompose_essential_matrix(E, shape, debug=True):
    U, _, Vt = svd(E)

    # rotation matrix with det = 1
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]

    # Correct rotations
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2

    # print(t)
    # t = np.array([100, 1, 1])  # translation vector
    # print(shape)
    # t[0] /= shape[0]
    # t[1] /= shape[1]
    t = 1 / t
    if debug:
        print(R1)
        print(R2)
        print(t)
    return R1, R2, t


def compute_rectification_homographies(R1, R2, t, K1, K2, debug=True):
    # Camera centers: c1 = origin, c2 = translation vector t
    c1 = np.zeros(3)
    c2 = t

    # Coordinate axes for rectification
    r1 = (c2 - c1) / np.linalg.norm(c2 - c1)
    r2 = np.cross([0, 0, 1], r1)
    r2 /= np.linalg.norm(r2)
    r3 = np.cross(r1, r2)
    # r3 = t
    # r3[2] = 1

    # Rectification rotation matrix
    R_rect = np.vstack([r1, r2, r3]).T

    if debug:
        print(R_rect)

    # Compute homographies
    H1 = K1 @ R_rect @ np.linalg.inv(K1)  # @ R1)
    H2 = K2 @ R_rect @ np.linalg.inv(K2)  # @ R2)

    if debug:
        print(H1)
        print(H2)

    return H1, H2


def apply_homography(img, H, debug=True):
    # Warp image with homography matrix H
    # This uses a meshgrid and affine mapping for each pixel

    h, w = img.shape[:2]
    coords = np.indices((w, h)).reshape(2, -1)
    coords_hom = np.vstack(
        (coords, np.ones((1, coords.shape[1])))
    )  # Homogeneous coordinates

    # Apply the homography
    transformed_coords = H @ coords_hom
    transformed_coords /= transformed_coords[2, :]  # Normalize by the third row
    transformed_coords = transformed_coords[:2].round().astype(int)
    if debug:
        print(transformed_coords)
    # Create an empty canvas for the warped image
    warped_img = np.zeros_like(img)
    x_valid = (0 <= transformed_coords[0, :]) & (transformed_coords[0, :] < w)
    y_valid = (0 <= transformed_coords[1, :]) & (transformed_coords[1, :] < h)
    valid = y_valid & x_valid

    warped_img[transformed_coords[1, valid], transformed_coords[0, valid]] = img[
        coords[1, valid], coords[0, valid]
    ]
    return warped_img


def run_rectification(left_image, right_image, debug=True):

    K1, K2, F, E = run_camera_calibration(debug=False)

    # Step 1: Decompose the essential matrix to get rotation and translation
    R1, R2, t = decompose_essential_matrix(E, left_image.shape, debug=debug)

    # Step 2: Compute the rectifying homographies
    H1, H2 = compute_rectification_homographies(R1, R2, t, K1, K2, debug=debug)

    # Step 3: Rectify both images by applying the homographies
    left_image_rectified = apply_homography(left_image, H1, debug=debug)
    right_image_rectified = apply_homography(right_image, H2, debug=debug)

    if debug:
        # display
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 2, 1)
        plt.imshow(left_image, cmap="gray")
        plt.title("Original Left Image")
        plt.subplot(2, 2, 2)
        plt.imshow(right_image, cmap="gray")
        plt.title("Original Right Image")
        plt.subplot(2, 2, 3)
        plt.imshow(left_image_rectified, cmap="gray")
        plt.title("Rectified Left Image")
        plt.subplot(2, 2, 4)
        plt.imshow(right_image_rectified, cmap="gray")
        plt.title("Rectified Right Image")
        plt.tight_layout()
        plt.show()

    return left_image_rectified, right_image_rectified


# left_image = Image.open("testleft.jpg").convert("L")
# right_image = Image.open("testright.jpg").convert("L")

# left_image = np.array(left_image)
# right_image = np.array(right_image)

# run_rectification(left_image, right_image, debug=True)
