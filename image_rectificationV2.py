from PIL import Image
import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from eight_point_algorithm import run_camera_calibration

left_image = Image.open("im0.png").convert("L")
right_image = Image.open("im1.png").convert("L")

left_image = np.array(left_image)
right_image = np.array(right_image)


def decompose_essential_matrix(E):
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

    t = np.array([12 / 100, 0, 0])  # translation vector
    # R1 = np.eye(3)  # no rotation
    # R1[0, 0] = 0.85
    # R2 = np.eye(3)  # no rotation
    R1[2, :] = 0
    R1[:, 2] = 0
    R1[2, 2] = -1
    R2[2, :] = 0
    R2[:, 2] = 0
    R2[2, 2] = 1
    print(R1)
    print(R2)
    return R1, R2, t


def compute_rectification_homographies(R1, R2, t, K1, K2):
    # Camera centers: c1 = origin, c2 = translation vector t
    c1 = np.zeros(3)
    c2 = t

    # Coordinate axes for rectification
    r1 = (c2 - c1) / np.linalg.norm(c2 - c1)
    r2 = np.cross([0, 0, 1], r1)
    r2 /= np.linalg.norm(r2)
    r3 = np.cross(r1, r2)

    # Rectification rotation matrix
    R_rect = np.vstack([r1, r2, r3]).T

    # Compute homographies
    H1 = K1 @ R_rect @ np.linalg.inv(K1 @ R1)
    H2 = K2 @ R_rect @ np.linalg.inv(K2 @ R2)

    return H1, H2


def apply_homography(img, H):
    # Warp image with homography matrix H
    # This uses a meshgrid and affine mapping for each pixel

    h, w = img.shape[:2]
    coords = np.indices((h, w)).reshape(2, -1)
    coords_hom = np.vstack(
        (coords, np.ones((1, coords.shape[1])))
    )  # Homogeneous coordinates

    # Apply the homography
    transformed_coords = H @ coords_hom
    transformed_coords /= transformed_coords[2, :]  # Normalize by the third row
    transformed_coords = transformed_coords[:2].round().astype(int)
    print(transformed_coords)
    # Create an empty canvas for the warped image
    warped_img = np.zeros_like(img)
    y_valid = (0 <= transformed_coords[0]) & (transformed_coords[0] < h)
    x_valid = (0 <= transformed_coords[1]) & (transformed_coords[1] < w)
    valid = y_valid & x_valid

    warped_img[transformed_coords[0, valid], transformed_coords[1, valid]] = img[
        coords[0, valid], coords[1, valid]
    ]
    return warped_img


K1, K2, F, E = run_camera_calibration(debug=False)

# Step 1: Decompose the essential matrix to get rotation and translation
R1, R2, t = decompose_essential_matrix(E)

# Step 2: Compute the rectifying homographies
H1, H2 = compute_rectification_homographies(R1, R2, t, K1, K2)

# Step 3: Rectify both images by applying the homographies
left_image_rectified = apply_homography(left_image, H1)
right_image_rectified = apply_homography(right_image, H2)


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
