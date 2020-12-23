import numpy as np
import cv2

from .image_processor import image_processor

class sobel_magnitude(image_processor):

    def __init__(self, sobel_kernel=3, mag_thresh=(0, 255)):
        self.__sobel_kernel = sobel_kernel
        self.__mag_thresh = mag_thresh
        
    def process_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.__sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.__sobel_kernel)
        sobel_mag = np.sqrt(sobelx**2 + sobely**2)
        
        scale_factor = np.max(sobel_mag)/255 
        sobel_mag = (sobel_mag/scale_factor).astype(np.uint8) 
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(sobel_mag)
        binary_output[(sobel_mag >= self.__mag_thresh[0]) & (sobel_mag <= self.__mag_thresh[1])] = 1
        
        return binary_output;
        