import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Helper Function
def calculate_SSD(matrix1, matrix2):
    return np.sum((matrix1 - matrix2) ** 2, axis=(-2, -1))

# Import images into CV
img_left = cv.imread("./test_images/view1.png")
img_right = cv.imread("./test_images/view5.png")

# Create grayscale versions of image
img_left_gray = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
img_right_gray = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)

# Initialize Variables
kernel_size = 5  # must be odd number
frame_thickness = (kernel_size - 1) // 2

img_height, img_width = img_left_gray.shape
disparity_map = np.zeros((img_height, img_width))

# Calculate all kernels
left_pixel_kernel_matrix = np.zeros((img_height - frame_thickness * 2, img_width - frame_thickness * 2, kernel_size, kernel_size))
right_pixel_kernel_matrix = np.zeros((img_height - frame_thickness * 2, img_width - frame_thickness * 2, kernel_size, kernel_size))

for y in range(frame_thickness, img_height - frame_thickness):
    for x in range(frame_thickness, img_width - frame_thickness):
        left_pixel_kernel_matrix[y - frame_thickness, x - frame_thickness] = img_left_gray[y - frame_thickness:y + frame_thickness + 1, x - frame_thickness:x + frame_thickness + 1]
        right_pixel_kernel_matrix[y - frame_thickness, x - frame_thickness] = img_right_gray[y - frame_thickness:y + frame_thickness + 1, x - frame_thickness:x + frame_thickness + 1]

# Vectorized SSD computation
for y in range(0, img_height - frame_thickness * 2):
    # Calculate SSDs for the whole row in the right image at once
    left_kernels = left_pixel_kernel_matrix[y]
    
    # SSD for each pixel's kernel against all kernels in the same row of the right image
    ssd = calculate_SSD(left_kernels[:, np.newaxis], right_pixel_kernel_matrix[y, :])
    
    # Find the best matching pixel for each pixel in the left image
    best_matching_pixel_xcoord = np.argmin(ssd, axis=1)
    disparity_map[y + frame_thickness, frame_thickness:img_width - frame_thickness] = np.abs(best_matching_pixel_xcoord - np.arange(frame_thickness, img_width - frame_thickness))

    print(y / (img_height - 2 * frame_thickness) * 100)  # Loading bar

fig, ax = plt.subplots(nrows=1, ncols=3)
ax[0].imshow(disparity_map, cmap='gray')
ax[1].imshow(img_left_gray, cmap='gray')
ax[2].imshow(img_right_gray, cmap='gray')
plt.show()

print(disparity_map)