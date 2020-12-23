import numpy as np
import cv2
import pickle
import glob
import os

from image_processors import camera_undistorter
from image_processors import hls_select
from image_processors import hsv_select
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
cv2.imwrite('test_debug_outputs/undistorted_image.jpg', undistorted_image)

images = glob.glob('test_images/test*.jpg')

for index, filename in enumerate(images):
    image = cv2.imread(filename)

    filename_base_path, extension = os.path.splitext(filename)
    filepath, filebase = os.path.split(filename_base_path)

    # sobel operator test
    orient = 'x'
    thresh_min = 20
    thresh_max = 100
    processor = sobel_operator(orient, thresh_min, thresh_max)
    sobel_x = processor.process_image(image)
    processed_name = os.path.join('test_debug_outputs', filebase+'sobel_x'+extension)
    cv2.imwrite(processed_name, sobel_x*255)

    orient = 'y'
    thresh_min = 20
    thresh_max = 100
    processor = sobel_operator(orient, thresh_min, thresh_max)
    sobel_y = processor.process_image(image)
    processed_name = os.path.join('test_debug_outputs', filebase+'sobel_x'+extension)
    cv2.imwrite(processed_name, sobel_y*255)

    # sobel magnitude test
    sobel_kernel = 3
    mag_thresh = (30, 100)
    processor = sobel_magnitude(sobel_kernel, mag_thresh)
    sobel_mag = processor.process_image(image)
    processed_name = os.path.join('test_debug_outputs', filebase+'sobel_magnitude'+extension)
    cv2.imwrite(processed_name, sobel_mag*255)

    # sobel direction test
    sobel_kernel = 15
    thresh = (0.7, 1.3)
    processor = sobel_direction(sobel_kernel, thresh)
    sobel_dir = processor.process_image(image)
    processed_name = os.path.join('test_debug_outputs', filebase+'sobel_direction'+extension)
    cv2.imwrite(processed_name, sobel_dir*255)

    # combine them
    combined = np.zeros_like(sobel_dir)
    combined[((sobel_x == 1) & (sobel_y == 1)) | ((sobel_mag == 1) & (sobel_dir == 1))] = 1
    processed_name = os.path.join('test_debug_outputs', filebase+'combined'+extension)
    cv2.imwrite(processed_name, combined*255)

    # hls select test
    channel = 's'
    thresh = (90, 255)
    processor = hls_select(channel, thresh)
    hls_select_image = processor.process_image(image)
    processed_name = os.path.join('test_debug_outputs', filebase+'hls_select'+extension)
    cv2.imwrite(processed_name, hls_select_image*255)

    # hls select test
    channel = 'v'
    thresh = (50, 255)
    processor = hsv_select(channel, thresh)
    hsv_select_image = processor.process_image(image)
    processed_name = os.path.join('test_debug_outputs', filebase+'hsv_select'+extension)
    cv2.imwrite(processed_name, hsv_select_image*255)
