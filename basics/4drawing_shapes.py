# Concept: Drawing Shapes & Text
# OpenCV can draw directly onto image arrays — rectangles, circles, lines,
# polygons, text, and even semi-transparent overlays.

import cv2
import numpy as np
from constants import input_img

img = cv2.imread(input_img)
img = cv2.resize(img, (500, 600))

# OpenCV uses BGR color order (not RGB) — so (0, 0, 255) is RED, not blue
RED    = (0, 0, 255)
GREEN  = (0, 255, 0)
BLUE   = (255, 0, 0)
YELLOW = (0, 255, 255)
WHITE  = (255, 255, 255)

# Rectangle: (image, top-left corner, bottom-right corner, color, thickness)
cv2.rectangle(img, (50, 50), (200, 200), BLUE, 3)

# Circle: (image, center, radius, color, thickness)
# Thickness = -1 means filled solid — useful for masks and markers
cv2.circle(img, (400, 150), 60, GREEN, -1)

# Line: (image, start point, end point, color, thickness)
cv2.line(img, (50, 250), (550, 250), RED, 5)

# Polygon: define corners as a NumPy array, then reshape for OpenCV's format
pts = np.array([[300, 350], [400, 450], [300, 550], [200, 450]], np.int32)
pts = pts.reshape((-1, 1, 2))   # OpenCV expects shape (N, 1, 2)
cv2.polylines(img, [pts], True, YELLOW, 3)  # True = closed shape (connects last to first)

# Text: (image, text, bottom-left of text, font, scale, color, thickness)
cv2.putText(img, "OpenCV Workshop", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)

# Semi-transparent overlay: draw on a copy, then blend it with the original
overlay = img.copy()
cv2.rectangle(overlay, (50, 400), (150, 550), (255, 0, 255), -1)
# addWeighted blends two images: result = alpha*src1 + beta*src2 + gamma
# Here: 20% overlay + 60% original = a faint transparent rectangle
cv2.addWeighted(overlay, 0.2, img, 0.6, 0, img)

cv2.imshow('Final Drawings', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
