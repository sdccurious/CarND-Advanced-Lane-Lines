import cv2
import pickle
import glob
import os

from image_processors import lane_finder
from image_processors import lane_analyzer

# load in camera calibration
camera_pickle = pickle.load( open( "camera_calibration_pickle.p", "rb" ) )
mtx = camera_pickle["mtx"]
dist = camera_pickle["dist"]

# load in perspective calibration
perspective_pickle = pickle.load( open( "perspective_calibration_pickle.p", "rb" ) )
M = perspective_pickle["M"]
Minv = perspective_pickle["Minv"]

# create an lane_finder
sobel_x_thresh = (12,255)
sobel_y_thresh = (12,255)
saturation_thresh = (100,255)

image_processor = lane_finder(mtx, dist, M, sobel_x_thresh, sobel_y_thresh, saturation_thresh)

nwindows = 9
margin = 100
minpix = 50
image_analyzer = lane_analyzer(nwindows, margin, minpix)

images = glob.glob('test_images/test*.jpg')

for index, filename in enumerate(images):
    image = cv2.imread(filename)
    processed_image = image_processor.process_image(image)
    analyzed_image = image_analyzer.process_image(processed_image)

    filename_base_path, extension = os.path.splitext(filename)
    filepath, filebase = os.path.split(filename_base_path)
    processed_name = os.path.join('output_images', filebase+'processed'+extension)
    cv2.imwrite(processed_name, analyzed_image)
