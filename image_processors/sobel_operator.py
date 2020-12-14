import numpy as np
import cv2

from .image_processor import image_processor

class sobel_operator(image_processor):

    def __init__(self, orient='x', thresh_min=0, thresh_max=255):
        self.__orient = orient
        self.__thresh_min = thresh_min
        self.__thresh_max = thresh_max
        
    def process_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        if self.__orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        if self.__orient == 'y':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)    
        
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
                
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= self.__thresh_min) & (scaled_sobel <= self.__thresh_max)] = 1
        
        return sxbinary;
        