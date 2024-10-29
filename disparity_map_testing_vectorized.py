import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Helper Function
def calculate_SSD(matrix1, matrix2):
    return np.sum((np.array(matrix1, dtype=np.float32) - np.array(matrix2, dtype=np.float32))**2)

# Import images into CV
img_left = cv.imread("./test_images/view1.png")
img_right = cv.imread("./test_images/view5.png")

# Create grayscale versions of image
img_left_gray = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
# img_left_gray = img_left_gray[150:351, 150:350]
img_right_gray = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)
# img_right_gray = img_right_gray[150:351, 150:350]

print(img_right.shape)
print(img_left.shape)

# Initialize Variables
kernel_size = 5 # must be odd number
frame_thickness = int((kernel_size-1)/2)

img_height = img_left_gray.shape[0]
img_width = img_left_gray.shape[1]

disparity_map = np.zeros((img_height, img_width))

# Calculate all kernels
left_pixel_kernel_matrix = np.zeros((img_height-frame_thickness*2, img_width-frame_thickness*2, kernel_size, kernel_size))
right_pixel_kernel_matrix = np.zeros((img_height-frame_thickness*2, img_width-frame_thickness*2, kernel_size, kernel_size))

for y in range(frame_thickness, img_height-frame_thickness):
    for x in range(frame_thickness, img_width-frame_thickness):
        # create intensity matrix
        left_pixel_kernel_matrix[y-frame_thickness,x-frame_thickness] = img_left_gray[y-frame_thickness:y+frame_thickness+1,x-frame_thickness:x+frame_thickness+1]
        right_pixel_kernel_matrix[y-frame_thickness,x-frame_thickness] = img_right_gray[y-frame_thickness:y+frame_thickness+1,x-frame_thickness:x+frame_thickness+1]

# for each pixel not on outside frame:
for y in range(0, img_height-frame_thickness*2):
    for x in range(0, img_width-frame_thickness*2):

        best_matching_pixel_xcoord = -1
        best_pixel_SSD = 200000
        
        
        # iterate through every pixel in right image in same row
        for x_r in range(0, img_width - frame_thickness*2):
           
            # find SSD between left pixel and right pixel
            kernel_SSD = calculate_SSD(left_pixel_kernel_matrix[y,x], right_pixel_kernel_matrix[y,x_r])
            
            # if pixel SSD is better than all previous pixels, select as "match" and set new threshold
            if kernel_SSD < best_pixel_SSD:
                best_matching_pixel_xcoord = x_r
                best_pixel_SSD = kernel_SSD

        # add disparity to disparity map
        if best_matching_pixel_xcoord != -1:
            disparity_map[y+frame_thickness,x+frame_thickness] = abs(best_matching_pixel_xcoord-x)
        # print(best_pixel_SSD)
    print(y/(img_width-2*frame_thickness) * 100) # used as a loading bar


fig, ax = plt.subplots(nrows=1, ncols=3)
ax[0].imshow(disparity_map, cmap='gray')
ax[1].imshow(img_left_gray, cmap='gray')
ax[2].imshow(img_right_gray, cmap='gray')
plt.show()


print(disparity_map)