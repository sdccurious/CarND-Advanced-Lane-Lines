from abc import ABCMeta, abstractmethod

class image_processor(metaclass=ABCMeta):

    @abstractmethod
    def process_image(self, image):
        pass
