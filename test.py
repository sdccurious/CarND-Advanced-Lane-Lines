import numpy as np
import cv2
import pickle

from image_processors import camera_undistorter
from image_processors import hls_select
from image_processors import sobel_direction
from image_processors import sobel_magnitude
from image_processors import sobel_operator

# load in camera calibration
dist_pickle = pickle.load( open( "camera_calibration_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# test camera calibration on an image
calibration_test_image = cv2.imread('camera_cal/calibration1.jpg')
processor = camera_undistorter(mtx, dist)
undistorted_image = processor.process_image(calibration_test_image)
cv2.imwrite('debug/undistorted_image.jpg', undistorted_image)

test1_image = cv2.imread('test_images/test1.jpg')

# sobel operator test
orient = 'x'
thresh_min = 20
thresh_max = 100
processor = sobel_operator(orient, thresh_min, thresh_max)
sobel_x = processor.process_image(test1_image)
cv2.imwrite('debug/sobel_x.jpg', sobel_x*255)

orient = 'y'
thresh_min = 20
thresh_max = 100
processor = sobel_operator(orient, thresh_min, thresh_max)
sobel_y = processor.process_image(test1_image)
cv2.imwrite('debug/sobel_y.jpg', sobel_y*255)

# sobel magnitude test
sobel_kernel = 3
mag_thresh = (30, 100)
processor = sobel_magnitude(sobel_kernel, mag_thresh)
sobel_mag = processor.process_image(test1_image)
cv2.imwrite('debug/sobel_magnitude.jpg', sobel_mag*255)

# sobel direction test
sobel_kernel = 15
thresh = (0.7, 1.3)
processor = sobel_direction(sobel_kernel, thresh)
sobel_dir = processor.process_image(test1_image)
cv2.imwrite('debug/sobel_direction.jpg', sobel_dir*255)

# combine them
combined = np.zeros_like(sobel_dir)
combined[((sobel_x == 1) & (sobel_y == 1)) | ((sobel_mag == 1) & (sobel_dir == 1))] = 1
cv2.imwrite('debug/sobel_combined.jpg', combined*255)

# hls select test
channel = 's'
thresh=(90, 255)
processor = hls_select(channel, thresh)
hls_select = processor.process_image(test1_image)
cv2.imwrite('debug/hls_select.jpg', hls_select*255)
