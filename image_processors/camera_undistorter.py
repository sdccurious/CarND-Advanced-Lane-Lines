import cv2

from .image_processor import image_processor

class camera_undistorter(image_processor):

    def __init__(self, mtx, dist):
        self.__mtx = mtx
        self.__dist = dist
        
    def process_image(self, image):
        output = cv2.undistort(image, self.__mtx, self.__dist, None, self.__mtx)
        
        return output;
        