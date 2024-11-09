from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage import transform

left_image = Image.open("im0.png").convert("L")
right_image = Image.open("im1.png").convert("L")

left_image_np = np.array(left_image)
right_image_np = np.array(right_image)

# camera parameters
K1 = np.array([[5255.409, 0, 658.094], [0, 5255.409, 948.705], [0, 0, 1]])
K2 = np.array([[5255.409, 0, 858.932], [0, 5255.409, 948.705], [0, 0, 1]])
baseline = 177.288 / 1000  # distance between 2 cameras
T = np.array([baseline, 0, 0])  # translation vector
R = np.eye(3)  # no rotation

# Projection matrices for both cameras ( how 3D points project onto 2D image planes)
P1 = np.dot(K1, np.hstack((R, np.zeros((3, 1)))))  # left
P2 = np.dot(K2, np.hstack((R, T.reshape(3, 1))))  # right


def warp_image(img, H):
    # applies a homography H to image
    h, w = img.shape
    y_idx, x_idx = np.indices((h, w))
    homogenous_coords = np.stack(
        (x_idx.ravel(), y_idx.ravel(), np.ones_like(x_idx.ravel()))
    )
    warped_coords = H @ homogenous_coords
    warped_coords /= warped_coords[2, :]
    x_warped, y_warped = warped_coords[0, :].reshape(h, w), warped_coords[1, :].reshape(
        h, w
    )
    warped_image = np.zeros_like(img)
    valid_mask = (x_warped >= 0) & (x_warped < w) & (y_warped >= 0) & (y_warped < h)
    warped_image[y_idx[valid_mask], x_idx[valid_mask]] = img[
        y_warped[valid_mask].astype(int), x_warped[valid_mask].astype(int)
    ]
    return warped_image


# rectification homographies based on projection matrices
H1 = np.linalg.inv(K1) @ P1[:, :3]
H2 = np.linalg.inv(K2) @ P2[:, :3]

# apply rectification homographies to both images
rectified_left = warp_image(left_image_np, H1)
rectified_right = warp_image(right_image_np, H2)

# display
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.imshow(left_image_np, cmap="gray")
plt.title("Original Left Image")
plt.subplot(2, 2, 2)
plt.imshow(right_image_np, cmap="gray")
plt.title("Original Right Image")
plt.subplot(2, 2, 3)
plt.imshow(rectified_left, cmap="gray")
plt.title("Rectified Left Image")
plt.subplot(2, 2, 4)
plt.imshow(rectified_right, cmap="gray")
plt.title("Rectified Right Image")
plt.tight_layout()
plt.show()
