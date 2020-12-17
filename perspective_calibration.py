import numpy as np
import cv2
import pickle

image = cv2.imread('test_images/straight_lines1.jpg')
x_size = image.shape[1]
y_size = image.shape[0]
x_mid = x_size / 2

circle_radius = 5
drawing_color = (255,0,0)
circle_thickness = -1
line_thickness = 5

# bottom of trapezoid
bottom_width = int(0.58 * x_size)
bottom_x_left = int(x_mid - (bottom_width / 2))
bottom_x_right = int(x_mid + (bottom_width / 2))

hood_location = 0.92
bottom_y = int(y_size * hood_location)

cv2.circle(image, (bottom_x_left, bottom_y), circle_radius, drawing_color, circle_thickness)
cv2.circle(image, (bottom_x_right, bottom_y), circle_radius, drawing_color, circle_thickness)

# top of trapezoid
top_width = int(0.12 * x_size)
top_x_left = int(x_mid - (top_width / 2))
top_x_right = int(x_mid + (top_width / 2))

horizon_location = 0.65
top_y = int(y_size * horizon_location)

cv2.circle(image, (top_x_left, top_y), circle_radius, drawing_color, circle_thickness)
cv2.circle(image, (top_x_right, top_y), circle_radius, drawing_color, circle_thickness)

#cv2.line(image, (bottom_x_left, bottom_y), (top_x_left, top_y), drawing_color, line_thickness)
#cv2.line(image, (bottom_x_right, bottom_y), (top_x_right, top_y), drawing_color, line_thickness)

cv2.imwrite('debug/perspective_image.jpg', image)

src = np.float32([(top_x_left, top_y), (top_x_right, top_y), (bottom_x_right, bottom_y), (bottom_x_left, bottom_y)])
dst = np.float32([(bottom_x_left, 0), (bottom_x_right, 0), (bottom_x_right, bottom_y), (bottom_x_left, bottom_y)])
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

image_size = (x_size, y_size)
warped = cv2.warpPerspective(image, M, image_size, flags=cv2.INTER_LINEAR)

cv2.imwrite('debug/perspective_image_warped.jpg', warped)

dist_pickle = {}
dist_pickle["M"] = M
dist_pickle["Minv"] = Minv
pickle.dump( dist_pickle, open( "perspective_calibration_pickle.p", "wb" ) )
