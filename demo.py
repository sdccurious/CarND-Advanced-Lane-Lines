import cv2
import pickle
import glob

from image_processors import lane_finder

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

images = glob.glob('test_images/test*.jpg')

for index, fileName in enumerate(images):
    image = cv2.imread(fileName)
    processed_image = image_processor.process_image(image)
    
    processed_name = 'output_images/processed' + str(index) + '.jpg'
    cv2.imwrite(processed_name, processed_image)
