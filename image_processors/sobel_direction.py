import numpy as np
import cv2

from .image_processor import image_processor

class sobel_direction(image_processor):

    def __init__(self, sobel_kernel=3, thresh=(0, np.pi/2)):
        self.__sobel_kernel = sobel_kernel
        self.__thresh = thresh
        
    def process_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.__sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.__sobel_kernel)
        
        abs_sobelx = np.abs(sobelx)
        abs_sobely = np.abs(sobely)
        gradient_direction = np.arctan2(abs_sobely, abs_sobelx)

        binary_output =  np.zeros_like(gradient_direction)
        binary_output[(gradient_direction >= self.__thresh[0]) & (gradient_direction <= self.__thresh[1])] = 1
        
        return binary_output;
        