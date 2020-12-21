import numpy as np
import cv2

from .image_processor import image_processor

class lane_analyzer(image_processor):

    def __init__(self, nwindows=9, margin=100, minpixels=50):
        self.__nwindows = nwindows
        self.__margin = margin
        self.__minpixels = minpixels
        self.__initialized = False
        self.__ploty = []
        self.__left_fitx = []
        self.__right_fitx = []

    def __find_lane_pixels(self, binary_warped):
        # Input needs to be a single channel image
        # Take a histogram of the bottom half of the image        
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
                                
        # Create an output image to draw on and visualize the result
        # Converts single channel image to 3 channel/BGR
        # out_img = np.dstack((binary_warped, binary_warped, binary_warped))

        out_img = np.zeros((binary_warped.shape[0], binary_warped.shape[1], 3), np.uint8)
        
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//self.__nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(self.__nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height

            win_xleft_low = leftx_current - self.__margin
            win_xleft_high = leftx_current + self.__margin
            win_xright_low = rightx_current - self.__margin
            win_xright_high = rightx_current + self.__margin
                        
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low,win_y_low), (win_xleft_high,win_y_high), (0,255,0), 2) 
            cv2.rectangle(out_img, (win_xright_low,win_y_low), (win_xright_high,win_y_high), (0,255,0), 2) 
            
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            if len(good_left_inds) > self.__minpixels:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.__minpixels:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        # cv2.imwrite('rectangle.jpg', out_img)

        return leftx, lefty, rightx, righty, out_img
        
    def __fit_polynomial(self, leftx, lefty, rightx, righty, image_with_windows):
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, image_with_windows.shape[0]-1, image_with_windows.shape[0] )
        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty

        ## Visualization ##
        # Colors in the left and right lane regions
        image_with_windows[lefty, leftx] = [255, 0, 0]
        image_with_windows[righty, rightx] = [0, 0, 255]
        
        # Plot polynomial
        for index, y in np.ndenumerate(ploty): 
            cv2.circle(image_with_windows, (int(left_fitx[index]), int(y)), radius=0, color=(0, 255, 255), thickness=-1)
            cv2.circle(image_with_windows, (int(right_fitx[index]), int(y)), radius=0, color=(0, 255, 255), thickness=-1)
            
        # Store for next time to start search
        self.__left_fitx = left_fitx
        self.__right_fitx = right_fitx

        return image_with_windows
        
    def __fit_poly(self, img_shape, leftx, lefty, rightx, righty):
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        # Generate x and y values for plotting
        ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        # Store for next time to start search and for plotting
        self.__ploty = ploty
        self.__left_fitx = left_fitx
        self.__right_fitx = right_fitx
        
        return left_fitx, right_fitx, ploty
        
    def __draw_poly(self, image, leftx, lefty, rightx, righty):
        image[lefty, leftx] = [255, 0, 0]
        image[righty, rightx] = [0, 0, 255]
        
        # Plot polynomial
        for index, y in np.ndenumerate(self.__ploty): 
            cv2.circle(image, (int(self.__left_fitx[index]), int(y)), radius=0, color=(0, 255, 255), thickness=-1)
            cv2.circle(image, (int(self.__right_fitx[index]), int(y)), radius=0, color=(0, 255, 255), thickness=-1)
            
        return image

    def __search_around_poly(self, binary_warped):
        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        left_lane_inds = ((nonzerox > (self.__left_fitx[0]*(nonzeroy**2) + self.__left_fitx[1]*nonzeroy + self.__left_fitx[2] - self.__margin))
                       & (nonzerox < (self.__left_fitx[0]*(nonzeroy**2) + self.__left_fitx[1]*nonzeroy + self.__left_fitx[2] + self.__margin)))
        right_lane_inds = ((nonzerox > (self.__right_fitx[0]*(nonzeroy**2) + self.__right_fitx[1]*nonzeroy + self.__right_fitx[2] - self.__margin))
                        & (nonzerox < (self.__right_fitx[0]*(nonzeroy**2) + self.__right_fitx[1]*nonzeroy + self.__right_fitx[2] + self.__margin)))
        
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit new polynomials
        left_fitx, right_fitx, ploty = self.__fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
        
        ## Visualization ##
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-self.__margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+self.__margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-self.__margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+self.__margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        
        # Plot polynomial
        for index, y in np.ndenumerate(ploty): 
            cv2.circle(image_with_windows, (int(left_fitx[index]), int(y)), radius=0, color=(0, 255, 255), thickness=-1)
            cv2.circle(image_with_windows, (int(right_fitx[index]), int(y)), radius=0, color=(0, 255, 255), thickness=-1)
        
        # Store for next time to start search
        self.__left_fitx = left_fitx
        self.__right_fitx = right_fitx
        
        return result
        
    def process_image(self, image):
        binary_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_shape = binary_image.shape
        
        if self.__initialized == False:
            leftx, lefty, rightx, righty, image_with_windows = self.__find_lane_pixels(binary_image)            
            left_fitx, right_fitx, ploty = self.__fit_poly(image_shape, leftx, lefty, rightx, righty)
            image_with_polynomial = self.__draw_poly(image_with_windows, leftx, lefty, rightx, righty)
            self.__initialized = True
        else:
            image_with_polynomial = self.__search_around_poly(binary_image)
        
        return image_with_polynomial;
