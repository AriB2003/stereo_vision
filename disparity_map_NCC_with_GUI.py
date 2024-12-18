import numpy as np
import cv2 as cv
import copy
import matplotlib.pyplot as plt
import matplotlib
import math

matplotlib.use("TKAgg")
from image_rectificationV2 import run_rectification

# Import images into CV
identifier = "classroom"
img_left = cv.imread(identifier + "left.jpg")
img_right = cv.imread(identifier + "right.jpg")
img_left = cv.resize(img_left, (0, 0), fx=0.25, fy=0.25)
img_right = cv.resize(img_right, (0, 0), fx=0.25, fy=0.25)

# Create grayscale versions of image
img_left_rect, img_right_rect = run_rectification(img_left, img_right, debug=False)
img_left_gray = cv.cvtColor(img_left_rect, cv.COLOR_RGB2GRAY)
img_right_gray = cv.cvtColor(img_right_rect, cv.COLOR_RGB2GRAY)
ksize = (5, 5)
sigmax = 0
img_left_gray = cv.GaussianBlur(img_left_gray, ksize, sigmax)
img_right_gray = cv.GaussianBlur(img_right_gray, ksize, sigmax)


# Helper Function
def calculate_SSD(matrix1, matrix2):  # use argmin for finding best fit
    return np.sum((matrix1 - matrix2) ** 2, axis=(-2, -1))


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
    denominator = np.sqrt(
        np.sum(norm1**2, axis=(-2, -1)) * np.sum(norm2**2, axis=(-2, -1))
    )

    # Prevent division by zero
    return np.where(denominator != 0, numerator / denominator, 0)


# img_left_gray = blur = cv.GaussianBlur(img_left_gray,(10,10),0)
# img_right_gray = blur = cv.GaussianBlur(img_right_gray,(10,10),0)

# Initialize Variables
kernel_size = 5  # must be odd number
frame_thickness = (kernel_size - 1) // 2

img_height, img_width = img_left_gray.shape
disparity_map = np.zeros((img_height, img_width))

# Calculate all kernels
left_pixel_kernel_matrix = np.zeros(
    (
        img_height - frame_thickness * 2,
        img_width - frame_thickness * 2,
        kernel_size,
        kernel_size,
    )
)
right_pixel_kernel_matrix = np.zeros(
    (
        img_height - frame_thickness * 2,
        img_width - frame_thickness * 2,
        kernel_size,
        kernel_size,
    )
)

for y in range(frame_thickness, img_height - frame_thickness):
    for x in range(frame_thickness, img_width - frame_thickness):
        left_pixel_kernel_matrix[y - frame_thickness, x - frame_thickness] = (
            img_left_gray[
                y - frame_thickness : y + frame_thickness + 1,
                x - frame_thickness : x + frame_thickness + 1,
            ]
        )
        right_pixel_kernel_matrix[y - frame_thickness, x - frame_thickness] = (
            img_right_gray[
                y - frame_thickness : y + frame_thickness + 1,
                x - frame_thickness : x + frame_thickness + 1,
            ]
        )

# Edge Filtering
t1 = 100
t2 = 120
ksize = (21, 21)
sigmax = 0
img_left_edges = cv.Canny(img_left, t1, t2)
img_left_edges[frame_thickness, :] = 255
img_left_edges[:, frame_thickness] = 255
img_left_edges[-frame_thickness - 1, :] = 255
img_left_edges[:, -frame_thickness - 1] = 255
cv.imshow("edges", img_left_edges)
img_left_edges = cv.GaussianBlur(img_left_edges, ksize, sigmax)
img_left_edges[img_left_edges > 0] = 255
cv.imshow("left", img_left_edges)
img_left_edges = img_left_edges > 0

img_right_edges = cv.Canny(img_right, t1, t2)
img_right_edges[frame_thickness, :] = 255
img_right_edges[:, frame_thickness] = 255
img_right_edges[-frame_thickness - 1, :] = 255
img_right_edges[:, -frame_thickness - 1] = 255
img_right_edges = cv.GaussianBlur(img_right_edges, ksize, sigmax)
img_right_edges[img_right_edges > 0] = 255
cv.imshow("right", img_right_edges)
img_right_edges = img_right_edges > 0

# cv.waitKey(-5)
# exit()

# Vectorized SSD computation
for y in range(0, img_height - frame_thickness * 2):
    # Calculate SSDs for the whole row in the right image at once
    left_kernels = left_pixel_kernel_matrix[
        y, img_left_edges[y, frame_thickness:-frame_thickness]
    ]
    # print(left_kernels.shape)
    # NCC for each pixel's kernel against all kernels in the same row of the right image
    ncc = calculate_SSD(
        left_kernels[:, np.newaxis],
        right_pixel_kernel_matrix[
            y, img_right_edges[y, frame_thickness:-frame_thickness]
        ],
    )
    # print(NCC)
    # Find the best matching pixel for each pixel in the left image
    best_matching_pixel_xcoord = np.argmin(ncc, axis=1)
    temp = np.arange(frame_thickness, img_width - frame_thickness)
    counter = 0
    inds = [
        i + frame_thickness
        for (i, val) in enumerate(img_right_edges[y, frame_thickness:-frame_thickness])
        if val == 1
    ]
    for i, v in enumerate(img_left_edges[y, frame_thickness:-frame_thickness]):
        if v:
            index = inds[best_matching_pixel_xcoord[counter]]
            temp[i] = index
            counter += 1
    best_matching_pixel_xcoord = temp

    lobf = np.arange(frame_thickness, img_width - frame_thickness)
    ssd = best_matching_pixel_xcoord - lobf
    m = np.mean(ssd)
    s = 2 * np.std(ssd)
    th1 = m - s
    th2 = m + s
    mask = np.logical_or(th1 > ssd, ssd > th2)
    best_matching_pixel_xcoord[mask] = lobf[mask]
    # print(np.abs(best_matching_pixel_xcoord - np.arange(frame_thickness, img_width - frame_thickness)))
    disparity_map[
        y + frame_thickness, frame_thickness : img_width - frame_thickness
    ] = np.abs(
        best_matching_pixel_xcoord
        - np.arange(frame_thickness, img_width - frame_thickness)
    )
    # print(disparity_map)
    print(y / (img_height - 2 * frame_thickness) * 100)  # Loading bar


# Implement the GUI
threshold = 255
blur_size = 1


def nothing(x):
    pass


cv.namedWindow("disp", cv.WINDOW_NORMAL)
cv.resizeWindow("disp", 600, 600)

cv.createTrackbar("threshold", "disp", 0, 255, nothing)
cv.createTrackbar("blur size", "disp", 1, 20, nothing)
cv.createTrackbar("gradients", "disp", 0, 1, nothing)

while True:
    # Updating the parameters based on the trackbar positions

    threshold = cv.getTrackbarPos("threshold", "disp")
    blur_size = cv.getTrackbarPos("blur size", "disp") * 2 - 1
    gradients = cv.getTrackbarPos("gradients", "disp")

    # Calculating stuff
    disparity_map_gui = copy.deepcopy(disparity_map)
    img_left_edges_gui = copy.deepcopy(img_left_edges)

    # Clean up image based on threshold
    for i in range(1, img_height - 1):
        for j in range(2, img_width - 2):
            if disparity_map[i, j] > threshold:
                disparity_map_gui[i, j] = 0
                img_left_edges_gui[i, j] = 0

    # Normalize
    disparity_map_scaled = cv.normalize(
        disparity_map_gui, None, 0, 255, cv.NORM_MINMAX
    )  # Normalize to 0-255
    disparity_map_uint8 = np.uint8(disparity_map_scaled)  # Convert to uint8

    # Add blur
    try:
        blurred_disparity_map = cv.medianBlur(disparity_map_uint8, blur_size, 0)
    except:
        blurred_disparity_map = cv.medianBlur(disparity_map_uint8, 1)

    # Add Gradients
    gradient_disparity_map = np.float64(blurred_disparity_map)
    if gradients:
        for y in range(gradient_disparity_map.shape[0]):
            last_edge_index = 0
            last_edge_value = gradient_disparity_map[y, 0]
            for x in range(gradient_disparity_map.shape[1]):
                if img_left_edges_gui[y, x]:
                    value_difference = gradient_disparity_map[y, x] - last_edge_value
                    linsp = np.linspace(
                        last_edge_value,
                        gradient_disparity_map[y, x],
                        num=x - last_edge_index + 1,
                    )
                    last_edge_value = gradient_disparity_map[y, x]
                    gradient_disparity_map[y, last_edge_index : x + 1] = linsp
                    last_edge_index = x

    # Normalize
    disparity_map_scaled = cv.normalize(
        gradient_disparity_map, None, 0, 255, cv.NORM_MINMAX
    )  # Normalize to 0-255
    disparity_map_uint8 = np.uint8(disparity_map_scaled)  # Convert to uint8

    # Displaying the disparity map
    cv.imshow("disp", disparity_map_uint8)

    # Close window using esc key
    if cv.waitKey(1) == 27:
        break


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
    theta_1 = math.pi / 2 - math.atan2(np.abs(d), focal_length) * np.sign(d)
    # print(theta_1)
    theta_2 = math.pi / 2 + math.atan2(np.abs(d_prime), focal_length) * np.sign(d_prime)
    # print(theta_2)
    try:
        x = float(
            (btwn_cameras * math.tan(theta_2)) / (math.tan(theta_1) + math.tan(theta_2))
        )
        y = float(x * math.tan(theta_1))
        return x, y
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
    theta = math.atan2(abs(d), f)
    # print(theta)
    # print(cos(theta))
    x = float(x_prime)
    y = float(y_prime * math.cos(theta))
    z = float(y_prime * math.sin(theta)) * np.sign(d)
    return x, y, z


f = 250  # focal length in pixels
btwn_cameras = 12


def calculate_depth_maps(processed_disparity_map, focal_length, btwn_cameras):

    f = focal_length
    img_height, img_width = processed_disparity_map.shape
    horiz_pixel_linspace = np.linspace(
        -(img_width - 1) / 2, (img_width - 1) / 2, img_width
    )
    vert_pixel_linspace = np.linspace(
        -(img_height - 1) / 2, (img_height - 1) / 2, img_height
    )

    depth_map_3D = np.zeros((img_height * img_width, 3))
    depth_map_2D = np.zeros((img_height, img_width))

    for i in range(img_height):
        for j in range(img_width):
            horiz_d = horiz_pixel_linspace[j]
            horiz_d_prime = horiz_pixel_linspace[j] - processed_disparity_map[i, j]
            x_prime, y_prime = calculate_pixel_coord(
                horiz_d, horiz_d_prime, f, btwn_cameras
            )
            vert_d = vert_pixel_linspace[i]
            try:
                x, y, z = transform_to_base_frame(x_prime, y_prime, f, vert_d)
                if 50 < np.abs(y) < 1000:
                    depth_map_3D[img_width * i + j, :] = [x, y, z]
                    depth_map_2D[i, j] = y
            except TypeError:
                pass

    return depth_map_2D, depth_map_3D


def plot_depth_map_2D(depth_map_2D):

    # # Cut off the left end of the depth map
    # processed_2D_depth = depth_map_2D[:,22:]
    processed_2D_depth = copy.deepcopy(depth_map_2D)

    # Process depth map for good visualization
    # threshold = 500
    # processed_2D_depth[processed_2D_depth > threshold] = 0
    processed_2D_depth = cv.normalize(processed_2D_depth, None, 0, 255, cv.NORM_MINMAX)
    processed_2D_depth = cv.GaussianBlur(processed_2D_depth, (5, 5), 2)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.imshow(processed_2D_depth, cmap="gray")

    def hover(event):
        if event.inaxes == ax:  # Check if the mouse is within the axis
            # Get the x and y pixel coordinates
            x, y = int(event.xdata), int(event.ydata)

            # Check if the coordinates are within the bounds of the image
            if (
                x >= 0
                and y >= 0
                and x < depth_map_2D.shape[1]
                and y < depth_map_2D.shape[0]
            ):
                # Get the pixel value at the current position
                pixel_value = depth_map_2D[y, x]

                # Display the pixel value on the plot
                ax.set_title(f"Pixel value at ({x}, {y}): {pixel_value}")
                fig.canvas.draw_idle()  # Update the plot immediately

    # Connect the hover function to the motion_notify_event
    fig.canvas.mpl_connect("motion_notify_event", hover)

    plt.show()


def plot_depth_map_3D(depth_map_3D):
    # Create the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Apply the function to the matrix
    depth_map_3D = remove_zero_rows(depth_map_3D)

    # Generate the values
    x_vals = depth_map_3D[::5, 0:1]  # Only using every 5th data point to reduce lag
    y_vals = depth_map_3D[::5, 1:2]
    z_vals = depth_map_3D[::5, 2:3]
    # Plot the values
    ax.scatter(x_vals, y_vals, z_vals, c="b", marker="o")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")

    plt.show()


depth_map_2D, depth_map_3D = calculate_depth_maps(disparity_map_gui, f, btwn_cameras)

plot_depth_map_2D(depth_map_2D)
