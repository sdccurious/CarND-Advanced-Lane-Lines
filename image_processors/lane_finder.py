import numpy as np
import cv2

from .camera_undistorter import camera_undistorter
from .perspective_warper import perspective_warper
from .image_processor import image_processor
from .sobel_operator import sobel_operator
from .hls_select import hls_select

class lane_finder(image_processor):

    def __init__(self, mtx, dst, M, sobel_x_thresh=(0,255), sobel_y_thresh=(0,255), saturation_thresh=(0,255)):
        self.__camera_undistorter = camera_undistorter(mtx, dst)
        self.__sobel_x = sobel_operator('x', sobel_x_thresh[0], sobel_x_thresh[1])
        self.__sobel_y = sobel_operator('y', sobel_y_thresh[0], sobel_y_thresh[1])
        self.__saturation = hls_select('s', saturation_thresh)
        self.__perspective_warper = perspective_warper(M)
        
    def process_image(self, image):
        undistorted_image = self.__camera_undistorter.process_image(image)
        sobel_x_image = self.__sobel_x.process_image(undistorted_image)
        sobel_y_image = self.__sobel_y.process_image(undistorted_image)
        saturation_image = self.__saturation.process_image(undistorted_image)
        
        combined = np.zeros_like(undistorted_image)
        combined[((sobel_x_image == 1) & (sobel_y_image == 1)) | (saturation_image == 1)] = 1
        warped = self.__perspective_warper.process_image(combined)
        
        return warped*255;
