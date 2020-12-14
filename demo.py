import cv2
import pickle

from image_processors import camera_undistorter

# load in camera calibration
dist_pickle = pickle.load( open( "calibration_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# test camera calibration on an image
img = cv2.imread('camera_cal/calibration1.jpg')
debug = True
processor = camera_undistorter(mtx, dist, debug)
processor.process_image(img)
