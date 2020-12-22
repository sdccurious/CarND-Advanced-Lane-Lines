import cv2
import glob
import os
import pickle

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
nwindows = 9
margin = 100
minpix = 50
ym_per_pix = 30/720
xm_per_pix = 3.7/700

images = glob.glob('test_images/test*.jpg')

for index, filename in enumerate(images):
    image = cv2.imread(filename)

    # create a new image lane processor for each image
    # the first call draws the sliding rectangles and subsequent calls fill in the image with a poly
    # so create a new one for each image to show the rectangles
    image_lane_processor = lane_finder(mtx, dist, M, Minv, sobel_x_thresh, sobel_y_thresh, saturation_thresh, nwindows, margin, minpix, ym_per_pix, xm_per_pix)
    image_with_lanes = image_lane_processor.process_image(image)

    filename_base_path, extension = os.path.splitext(filename)
    filepath, filebase = os.path.split(filename_base_path)
    processed_name = os.path.join('output_images', filebase+'processed'+extension)
    cv2.imwrite(processed_name, image_with_lanes)

video_lane_processor = lane_finder(mtx, dist, M, Minv, sobel_x_thresh, sobel_y_thresh, saturation_thresh, nwindows, margin, minpix, ym_per_pix, xm_per_pix)
project_video_output = ('output_images/project_video_processed.mp4')
project_video = VideoFileClip("project_video.mp4")
project_video_clip = project_video.fl_image(video_lane_processor.process_image)
project_video_clip.write_videofile(project_video_output, audio=False)
