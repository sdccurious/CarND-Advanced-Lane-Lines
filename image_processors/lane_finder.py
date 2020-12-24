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

    # This is the main image processing class for finding lanes.  All the various image processors are instantiated within this class.
    # The logic for handling/combining the outputs of the various member processor classes is handled here.
    # The logic for analayzing the lane which is mostly math is handled in the lane_analyzer instance.
    def __init__(self, mtx, dst, M, Minv, sobel_x_thresh, sobel_y_thresh, saturation_thresh, nwindows, margin, minpix, ym_per_pix, xm_per_pix):
        self.__camera_undistorter = camera_undistorter(mtx, dst)
        self.__sobel_x = sobel_operator('x', sobel_x_thresh[0], sobel_x_thresh[1])
        self.__sobel_y = sobel_operator('y', sobel_y_thresh[0], sobel_y_thresh[1])
        self.__saturation = hls_select('s', saturation_thresh)
        self.__perspective_warper = perspective_warper(M)
        self.__perspective_unwarper = perspective_warper(Minv)
        self.__lane_analyzer = lane_analyzer(nwindows, margin, minpix, ym_per_pix, xm_per_pix)
        
    def process_image(self, image, debug=False):
        # Prepare the image isolating the lane pixels.
        undistorted_image = self.__camera_undistorter.process_image(image)
        sobel_x_image = self.__sobel_x.process_image(undistorted_image)
        sobel_y_image = self.__sobel_y.process_image(undistorted_image)
        saturation_image = self.__saturation.process_image(undistorted_image)

        combined = np.zeros_like(undistorted_image)
        # We want the pixels where the sobel in X AND Y are active OR the saturation channel is active
        combined[((sobel_x_image == 1) & (sobel_y_image == 1)) | (saturation_image == 1)] = 1
        # Put the image into plan view for the lane_analyzer
        warped = self.__perspective_warper.process_image(combined)

        analyzed_image, average_curvature, offset = self.__lane_analyzer.process_image(warped)

        # Unwarp the image for overlaying and combine with the OG image
        analyzed_image_unwarped = self.__perspective_unwarper.process_image(analyzed_image)
        image_with_lanes = cv2.addWeighted(image, 1.0, analyzed_image_unwarped, 1.0, 0)

        # Add the text for displaying the curvature and offset
        text = 'Curvature (m): ' + str(round(average_curvature, 3))
        bottomLeftCornerOfText = (10,100)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        fontColor = (255,255,255)
        lineType = 2
        image_with_lanes_and_text = cv2.putText(image_with_lanes, text, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

        if offset <=0:
            side = 'right'
        else:
            side = 'left'

        text = 'Offset (m): ' + str(round(offset, 3)) + ' ' + side
        bottomLeftCornerOfText = (10,200)
        image_with_lanes_and_text = cv2.putText(image_with_lanes, text, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

        # This is just to create sample images for the report.
        if debug == True:
            cv2.imwrite('test_debug_outputs/undistorted_sample.jpg', undistorted_image)
            cv2.imwrite('test_debug_outputs/sobel_x_sample.jpg', sobel_x_image*255)
            cv2.imwrite('test_debug_outputs/sobel_y_sample.jpg', sobel_y_image*255)
            cv2.imwrite('test_debug_outputs/saturation_sample.jpg', saturation_image*255)
            cv2.imwrite('test_debug_outputs/binary_sample.jpg', combined*255)
            cv2.imwrite('test_debug_outputs/warped_sample.jpg', warped*255)
            cv2.imwrite('test_debug_outputs/analyzed_sample.jpg', analyzed_image)
            cv2.imwrite('test_debug_outputs/analyzed_unwarped_sample.jpg', analyzed_image_unwarped)

        return image_with_lanes_and_text;
