import numpy as np
import cv2

from .image_processor import image_processor

class hsv_select(image_processor):

    def __init__(self, channel='V', thresh=(0, 255)):
        self.__channel = channel.upper()
        self.__thresh = thresh
        
    def process_image(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        if self.__channel == 'H':
            channel = 0
        elif self.__channel == "S":
            channel = 1
        elif self.__channel == "V":
            channel = 2
        else:
            raise LookupError
        
        data = hsv[:,:,channel]
    
        binary = np.zeros_like(data)
        binary[(data > self.__thresh[0]) & (data <= self.__thresh[1])] = 1
    
        return binary
    