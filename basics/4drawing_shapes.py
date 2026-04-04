import cv2
import numpy as np
    

img = cv2.imread('input.png')
img = cv2.resize(img, (500, 600))

# openCV uses BGR, not RGB
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)

# Rectangle: (image, top-left, bottom-right, color, thickness)
cv2.rectangle(img, (50, 50), (200, 200), BLUE, 3)

# Circle: (image, center, radius, color, thickness) 
# Use -1 for thickness to FILL the shape
cv2.circle(img, (400, 150), 60, GREEN, -1)

# Line: (image, start, end, color, thickness)
cv2.line(img, (50, 250), (550, 250), RED, 5)

# POLYGONS (Polyshape)
# Defining points for a diamond/star shape
pts = np.array([[300, 350], [400, 450], [300, 550], [200, 450]], np.int32)
# Reshape points for OpenCV (required for polylines)
pts = pts.reshape((-1, 1, 2))
cv2.polylines(img, [pts], True, YELLOW, 3)

# TEXT
# (image, text, position, font, scale, color, thickness)
cv2.putText(img, "OpenCV Workshop", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)

# Semi transparent overlay
# Create a copy, draw a filled shape, then blend it
overlay = img.copy()
cv2.rectangle(overlay, (50, 400), (150, 550), (255, 0, 255), -1)
# alpha = transparency of original, beta = transparency of overlay
cv2.addWeighted(overlay, 0.2, img, 0.6, 0, img)

# 7. DISPLAY
cv2.imshow('Final Drawings', img)
cv2.waitKey(0)
cv2.destroyAllWindows()