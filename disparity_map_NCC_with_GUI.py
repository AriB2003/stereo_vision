import numpy as np
import cv2 as cv
import copy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')

# Helper Function
def calculate_SSD(matrix1, matrix2):
    return np.sum((matrix1 - matrix2) ** 2, axis=(-2, -1))

def calculate_NCC(matrix1, matrix2):
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
img_left = cv.imread("./test_images/bview1.png")
img_right = cv.imread("./test_images/bview5.png")

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
    # NCC for each pixel's kernel against all kernels in the same row of the right image
    ncc = calculate_NCC(left_kernels[:, np.newaxis], right_pixel_kernel_matrix[y,:])
    # print(NCC)
    # Find the best matching pixel for each pixel in the left image
    best_matching_pixel_xcoord = np.argmax(ncc, axis=1)
    # print(np.abs(best_matching_pixel_xcoord - np.arange(frame_thickness, img_width - frame_thickness)))
    disparity_map[y + frame_thickness, frame_thickness:img_width - frame_thickness] = np.abs(best_matching_pixel_xcoord - np.arange(frame_thickness, img_width - frame_thickness))
    # print(disparity_map)
    print(y / (img_height - 2 * frame_thickness) * 100)  # Loading bar


# Implement the GUI
threshold = 255
blur_size = 1

def nothing(x):
    pass

cv.namedWindow('disp',cv.WINDOW_NORMAL)
cv.resizeWindow('disp',600,600)

cv.createTrackbar('threshold', 'disp', 0, 255, nothing)
cv.createTrackbar('blur size', 'disp', 1, 20, nothing)
 
while True:
    # Updating the parameters based on the trackbar positions
    
    threshold = cv.getTrackbarPos('threshold','disp')
    blur_size = cv.getTrackbarPos('blur size', 'disp') * 2 - 1
    
    # Calculating stuff
    disparity_map_gui = copy.deepcopy(disparity_map)

    for i in range(1, img_height - 1):
        for j in range(2, img_width - 2):
            if disparity_map[i,j] > threshold:
                disparity_map_gui[i,j] = 0

    disparity_map_scaled = cv.normalize(disparity_map_gui, None, 0, 255, cv.NORM_MINMAX)  # Normalize to 0-255
    disparity_map_uint8 = np.uint8(disparity_map_scaled)  # Convert to uint8

    try: 
        blurred_disparity_map = cv.medianBlur(disparity_map_uint8, blur_size)
    except: 
        blurred_disparity_map = cv.medianBlur(disparity_map_uint8, 1)

    # Displaying the disparity map
    cv.imshow("disp",blurred_disparity_map)

    # Close window using esc key
    if cv.waitKey(1) == 27:
        break    