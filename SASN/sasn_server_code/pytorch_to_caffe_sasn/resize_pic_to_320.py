import numpy
import cv2

rgb_image = cv2.imread('/path/to/your/image')
rgb_image = cv2.resize(rgb_image, (320, 320))
cv2.imwrite('image1.png', rgb_image)
