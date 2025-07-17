import cv2
import numpy as np

print("OpenCV version:", cv2.__version__)

# Create a blank image (white square)
img = np.zeros((500, 500, 3), dtype=np.uint8) + 255 

cv2.imshow('Test Window', img)
print("Window should be displayed. Press any key to close.")
cv2.waitKey(0) # Wait indefinitely until a key is pressed
cv2.destroyAllWindows()
print("Window closed.")