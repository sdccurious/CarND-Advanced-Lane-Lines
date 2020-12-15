import cv2

from .image_processor import image_processor

class perspective_warper(image_processor):

    def __init__(self, M):
        self.__M = M
        self.__dist = dist
        
    def process_image(self, image):
        image_size = (image.shape[1], image.shape[0]) 
    
        output = cv2.warpPerspective(image, self.__M, image_size, flags=cv2.INTER_LINEAR)
        
        return output;
        