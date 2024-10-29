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
img_left_gray = img_left_gray[150:351, 150:350]
img_right_gray = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)
img_right_gray = img_right_gray[150:351, 150:350]

print(img_right.shape)
print(img_left.shape)

# Initialize Variables
img_width = img_left_gray.shape[1]
img_height = img_left_gray.shape[0]
print(img_height)

disparity_map = np.zeros((img_height, img_width))

# DISPARITY MAPPING USING WINDOW SSD
# note: images are in y,x format, 

# for each pixel not on outside frame:
for y in range(2, img_height-2):
    for x in range(2, img_width-2):
        # create intensity matrix
        left_pixel_kernel = img_left_gray[y-2:y+3,x-2:x+3] # 3x3

        best_matching_pixel_xcoord = -1
        best_pixel_SSD = 200000

        # iterate through every pixel in right image in same row
        for x_r in range(2, img_width-2):
            # current pixel that is being compared
            current_right_pixel = img_right_gray[y, x_r]
            
            # create intensity matrix
            right_pixel_kernel = img_right_gray[y-2:y+3,x_r-2:x_r+3] # 3x3
            
            # find SSD between left pixel and right pixel
            kernel_SSD = calculate_SSD(left_pixel_kernel, right_pixel_kernel)
            
            # if pixel SSD is better than all previous pixels, select as "match" and set new threshold
            if kernel_SSD < best_pixel_SSD:
                best_matching_pixel_xcoord = x_r
                best_pixel_SSD = kernel_SSD

        # add disparity to disparity map
        if best_matching_pixel_xcoord != -1:
            disparity_map[y,x] = abs(best_matching_pixel_xcoord-x)
        # print(best_pixel_SSD)
    print(y/(img_width-2) * 100) # used as a loading bar


max = 0

for i in range(50):
    for j in range(50):
        if np.array(disparity_map)[i,j] > max:
            max = np.array(disparity_map)[i,j]

print(max)

fig, ax = plt.subplots(nrows=1, ncols=3)
ax[0].imshow(disparity_map, cmap='gray')
ax[1].imshow(img_left_gray, cmap='gray')
ax[2].imshow(img_right_gray, cmap='gray')
plt.show()


print(disparity_map)