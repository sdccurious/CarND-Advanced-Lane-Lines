import numpy as np
import cv2

from .camera_undistorter import camera_undistorter
from .hls_select import hls_select
from .hsv_select import hsv_select
from .image_processor import image_processor
from .lane_analyzer import lane_analyzer
from .perspective_warper import perspective_warper
from .sobel_operator import sobel_operator

class lane_finder(image_processor):

    def __init__(self, mtx, dst, M, Minv, sobel_x_thresh, sobel_y_thresh, saturation_thresh, value_thresh, nwindows, margin, minpix, ym_per_pix):
        self.__camera_undistorter = camera_undistorter(mtx, dst)
        self.__sobel_x = sobel_operator('x', sobel_x_thresh[0], sobel_x_thresh[1])
        self.__sobel_y = sobel_operator('y', sobel_y_thresh[0], sobel_y_thresh[1])
        self.__saturation = hls_select('s', saturation_thresh)
        self.__value = hsv_select('v', value_thresh)
        self.__perspective_warper = perspective_warper(M)
        self.__perspective_unwarper = perspective_warper(Minv)
        self.__lane_analyzer = lane_analyzer(nwindows, margin, minpix, ym_per_pix)
        
    def process_image(self, image):
        undistorted_image = self.__camera_undistorter.process_image(image)
        sobel_x_image = self.__sobel_x.process_image(undistorted_image)
        sobel_y_image = self.__sobel_y.process_image(undistorted_image)
        saturation_image = self.__saturation.process_image(undistorted_image)
        value_image = self.__value.process_image(undistorted_image)
        
        combined = np.zeros_like(undistorted_image)
        combined[((sobel_x_image == 1) & (sobel_y_image == 1)) | ((saturation_image == 1) & (value_image == 1))] = 1
        warped = self.__perspective_warper.process_image(combined)

        analyzed_image, left_curvature, right_curvature = self.__lane_analyzer.process_image(warped)
        analyzed_image_unwarped= self.__perspective_unwarper.process_image(analyzed_image)

        image_with_lanes = cv2.addWeighted(image, 1.0, analyzed_image_unwarped, 1.0, 0)

        text = 'Left curvature: ' + str(left_curvature)
        bottomLeftCornerOfText = (10,100)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        fontColor = (255,255,255)
        lineType = 2
        image_with_lanes_and_text = cv2.putText(image_with_lanes, text, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

        text = 'Right curvature: ' + str(right_curvature)
        bottomLeftCornerOfText = (10,200)
        image_with_lanes_and_text = cv2.putText(image_with_lanes_and_text, text, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
        return image_with_lanes_and_text;
