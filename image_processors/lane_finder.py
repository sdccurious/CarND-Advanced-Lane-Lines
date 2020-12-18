import numpy as np
import cv2

from .image_processor import image_processor
from .sobel_operator import sobel_operator
from .hls_select import hls_select

class lane_finder(image_processor):

    def __init__(self, sobel_x_thresh=(0,255), sobel_y_thresh=(0,255), saturation_thresh=(0,255)):
        self.__sobel_x = sobel_operator('x', sobel_x_thresh[0], sobel_x_thresh[1])
        self.__sobel_y = sobel_operator('y', sobel_y_thresh[0], sobel_y_thresh[1])
        self.__saturation = hls_select('s', saturation_thresh)
        
    def process_image(self, image):
        sobel_x_image = self.__sobel_x.process_image(image)
        sobel_y_image = self.__sobel_y.process_image(image)
        saturation_image = self.__saturation.process_image(image)
        
        combined = np.zeros_like(image)
        combined[((sobel_x_image == 1) & (sobel_y_image == 1)) | (saturation_image == 1)] = 1
        
        return combined*255;
        