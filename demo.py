import cv2
import glob
import os
import pickle
import sys

from moviepy.editor import VideoFileClip

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
value_thresh = (50,255)
nwindows = 9
margin = 100
minpix = 50
ym_per_pix = 30/720 # meters per pixel in y dimension

lane_processor = lane_finder(mtx, dist, M, Minv, sobel_x_thresh, sobel_y_thresh, saturation_thresh, value_thresh, nwindows, margin, minpix, ym_per_pix)

images = glob.glob('test_images/test*.jpg')

for index, filename in enumerate(images):
    image = cv2.imread(filename)
    image_with_lanes = lane_processor.process_image(image)

    filename_base_path, extension = os.path.splitext(filename)
    filepath, filebase = os.path.split(filename_base_path)
    processed_name = os.path.join('output_images', filebase+'processed'+extension)
    cv2.imwrite(processed_name, image_with_lanes)

#sys.exit()
project_video_output = ('output_images/project_video_processed.mp4')
project_video = VideoFileClip("project_video.mp4")
project_video_clip = project_video.fl_image(lane_processor.process_image)
project_video_clip.write_videofile(project_video_output, audio=False)
