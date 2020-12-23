import numpy as np
import cv2

from .image_processor import image_processor

class hls_select(image_processor):

    def __init__(self, channel='S', thresh=(0, 255)):
        self.__channel = channel.upper()
        self.__thresh = thresh
        
    def process_image(self, image):
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        
        if self.__channel == 'H':
            channel = 0
        elif self.__channel == "L":
            channel = 1
        elif self.__channel == "S":
            channel = 2
        else:
            raise LookupError
        
        data = hls[:,:,channel]
    
        binary = np.zeros_like(data)
        binary[(data > self.__thresh[0]) & (data <= self.__thresh[1])] = 1
    
        return binary
    