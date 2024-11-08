import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')

# Helper Function
def calculate_SSD(matrix1, matrix2): # use argmin for finding best fit
    return np.sum((matrix1 - matrix2) ** 2, axis=(-2, -1))

def calculate_NCC(matrix1, matrix2): # use argmax for finding best fit
    # Ensure matrices are numpy arrays for compatibility
    matrix1 = np.asarray(matrix1)
    matrix2 = np.asarray(matrix2)
    
    # Calculate mean along the last two axes for each matrix in the batch
    mean1 = np.mean(matrix1, axis=(-2, -1), keepdims=True)
    mean2 = np.mean(matrix2, axis=(-2, -1), keepdims=True)
    
    # Subtract means and compute normalized matrices
    norm1 = matrix1 - mean1
    norm2 = matrix2 - mean2
    
    # Compute numerator and denominator in a batch-wise manner
    numerator = np.sum(norm1 * norm2, axis=(-2, -1))
    denominator = np.sqrt(np.sum(norm1 ** 2, axis=(-2, -1)) * np.sum(norm2 ** 2, axis=(-2, -1)))
    
    # Prevent division by zero
    return np.where(denominator != 0, numerator / denominator, 0)

# Import images into CV
img_left = cv.imread("./test_images/view1.png")
img_right = cv.imread("./test_images/view5.png")

# Create grayscale versions of image
img_left_gray = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
img_right_gray = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)

# img_left_gray = blur = cv.GaussianBlur(img_left_gray,(10,10),0)
# img_right_gray = blur = cv.GaussianBlur(img_right_gray,(10,10),0)

# Initialize Variables
kernel_size = 7 # must be odd number
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
    # print(left_kernels.shape)
    # SSD for each pixel's kernel against all kernels in the same row of the right image
    ssd = calculate_SSD(left_kernels[:, np.newaxis], right_pixel_kernel_matrix[y,:])
    print(ssd)
    # Find the best matching pixel for each pixel in the left image
    best_matching_pixel_xcoord = np.argmin(ssd, axis=1)
    # print(np.abs(best_matching_pixel_xcoord - np.arange(frame_thickness, img_width - frame_thickness)))
    disparity_map[y + frame_thickness, frame_thickness:img_width - frame_thickness] = np.abs(best_matching_pixel_xcoord - np.arange(frame_thickness, img_width - frame_thickness))
    # print(disparity_map)
    print(y / (img_height - 2 * frame_thickness) * 100)  # Loading bar


# for i in range(1, img_height - 1):
#     for j in range(2, img_width - 2):
#         if disparity_map[i,j] > 75:
#             disparity_map[i,j] = 0

threshold = 75
disparity_map[disparity_map > threshold] = 0


disparity_map_scaled = cv.normalize(disparity_map, None, 0, 255, cv.NORM_MINMAX)  # Normalize to 0-255
disparity_map_uint8 = np.uint8(disparity_map_scaled)  # Convert to uint8
blurred_disparity_map = cv.medianBlur(disparity_map_uint8, 5)
# blurred_disparity_map = cv.GaussianBlur(disparity_map_uint8, (5,5), 1)

fig, ax = plt.subplots(nrows=1, ncols=3)
ax[0].imshow(disparity_map_uint8, cmap='gray')
ax[1].imshow(img_left_gray, cmap='gray')
ax[2].imshow(img_right_gray, cmap='gray')
plt.show()

print(disparity_map)
print(disparity_map[216,270])