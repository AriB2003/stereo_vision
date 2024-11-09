import numpy as np
import cv2 as cv
import copy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')
from image_rectificationV2 import run_rectification

# Import images into CV
img_left = cv.imread("./test_images/bview1.png")
img_right = cv.imread("./test_images/bview5.png")

# Create grayscale versions of image
img_left_gray, img_right_gray = run_rectification(img_left, img_right, debug=False)






























# Helper Function
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

# img_left_gray = blur = cv.GaussianBlur(img_left_gray,(10,10),0)
# img_right_gray = blur = cv.GaussianBlur(img_right_gray,(10,10),0)

# Initialize Variables
kernel_size = 5 # must be odd number
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

    # Clean up image based on threshold
    for i in range(1, img_height - 1):
        for j in range(2, img_width - 2):
            if disparity_map[i,j] > threshold:
                disparity_map_gui[i,j] = 0

    # Normalize and add blur
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

# Processed disparity map for use in depth map visualizations
# We have to create this separately because normalization used to improve GUI usability distorts our results
disparity_map_processed = copy.deepcopy(disparity_map)
disparity_map_processed[disparity_map_processed > threshold] = 0
disparity_map_processed = cv.medianBlur(np.uint8(disparity_map), blur_size)

# disparity_map = 370x420 matrix
import numpy as np
import math
from mpl_toolkits.mplot3d import axes3d

def calculate_pixel_coord(d, d_prime, focal_length, btwn_cameras):
    """
    Calculates the coordinates of a pixel given its position within two images
    taken from a stereo camera.

    Args:
        d: horizontal distance from center of frame in first image (in pixels)
        d_prime: horizontal distance from center of frame in first image (in pixels)
        focal_length: focal length of the camera used (in pixels)
        btwn_cameras: horizontal distance between the left and right cameras 
        (in same units as coordinate frame output)

    Returns:
        x: x coordinate in plane
        y: y coordinate in plane
    """
    theta_1 = math.pi/2 - np.atan2(np.abs(d), focal_length) * np.sign(d)
    # print(theta_1)
    theta_2 = math.pi/2 + np.atan2(np.abs(d_prime), focal_length) * np.sign(d_prime)
    # print(theta_2)
    try:
        x = float((btwn_cameras*math.tan(theta_2))/(math.tan(theta_1)+math.tan(theta_2)))
        y = float(x * math.tan(theta_1))
        return x,y
    except ZeroDivisionError:
        return None, None

def transform_to_base_frame(x_prime, y_prime, f, d):
    """
    Transforms a pixel coordinate from its epipolar plane coordinates to base
    frame coordinates given its height within the image.

    Args:
        x_prime: epipolar plane x_coordinate
        y_prime: epipolar plane y_coordinate
        f: focal length of the camera used (in pixels)
        d: distance above the center of the image (in pixels)

    Returns: 
        x,y,z: x, y, and z coordinates in base frame.
    """
    theta = math.atan2(abs(d),f)
    # print(theta)
    # print(cos(theta))
    x = float(x_prime)
    y = float(y_prime * math.cos(theta))
    z = float(y_prime * math.sin(theta)) * np.sign(d)
    return x,y,z

f = 250 # focal length in pixels
btwn_cameras = 12

horiz_pixel_linspace = np.linspace(-(img_width-1)/2, (img_width-1)/2, img_width)
vert_pixel_linspace = np.linspace(-(img_height-1)/2, (img_height-1)/2, img_height)

depth_map_3D = np.zeros((img_height*img_width,3))

depth_map_2D = np.zeros((img_height, img_width))

for i in range(img_height):
    for j in range(img_width):
        horiz_d = horiz_pixel_linspace[j]
        horiz_d_prime = horiz_pixel_linspace[j] - disparity_map_processed[i,j]
        x_prime, y_prime = calculate_pixel_coord(horiz_d, horiz_d_prime, f, btwn_cameras)
        vert_d = vert_pixel_linspace[i]
        try:
            x, y, z = transform_to_base_frame(x_prime, y_prime, f, vert_d)
            if np.abs(x) < 750 and 10 < np.abs(y) < 750 and np.abs(z) < 750:               
                depth_map_3D[img_width*i + j, :] = [x,y,z]
                depth_map_2D[i,j] = y
        except TypeError:
            pass

depth_map_2D = cv.normalize(depth_map_2D, None, 0, 255, cv.NORM_MINMAX)
threshold = 35
depth_map_2D[depth_map_2D > threshold] = 0
depth_map_2D = cv.normalize(depth_map_2D, None, 0, 255, cv.NORM_MINMAX)

fig = plt.figure()
ax = fig.add_subplot()
ax.imshow(depth_map_2D, cmap='gray')
plt.show()

# Create the figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def remove_zero_rows(matrix):
    # Keep rows where at least one element is not zero
    return matrix[~np.all(matrix == 0, axis=1)]

# Apply the function to the matrix
X_iso = remove_zero_rows(depth_map_3D)

# Generate the values
x_vals = X_iso[::5, 0:1]
y_vals = X_iso[::5, 1:2]
z_vals = X_iso[::5, 2:3]
# Plot the values
ax.scatter(x_vals, y_vals, z_vals, c = 'b', marker='o')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

plt.show()