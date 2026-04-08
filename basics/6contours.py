# Concept: Edge Detection & Contours
# Edges are boundaries where pixel intensity changes sharply.
# Contours are the outlines of objects — think of them as connect-the-dots along edges.
# This file finds shapes in an image and draws bounding boxes around them.

import cv2
import numpy as np
from constants import contour_input_img

img = cv2.imread(contour_input_img)   # Use an image with clear, distinct shapes
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- EDGE DETECTION ---
# Sobel: detects edges in one direction (here: horizontal, dx=1, dy=0)
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)

# Canny: a smarter edge detector — applies smoothing, gradients, and hysteresis
# (50, 150) are the low and high thresholds for what counts as an edge
canny = cv2.Canny(gray, 50, 150)

# --- CONTOURS ---
# findContours scans the binary (Canny) image and traces outlines of white regions
# RETR_TREE: finds all contours and organizes parent-child relationships (nested shapes)
# CHAIN_APPROX_SIMPLE: compresses straight lines to just their endpoints (saves memory)
contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 50:               # Skip tiny noise blobs (area < 50 pixels is probably junk)

        peri = cv2.arcLength(cnt, True)                    # Perimeter of the contour
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True) # Approximate the shape with fewer points

        # Bounding rectangle: axis-aligned box that fits around the contour
        x, y, w, h = cv2.boundingRect(cnt)

        # Minimum enclosing circle: smallest circle that fits around the contour
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)

        # Draw the contour outline, bounding box, and enclosing circle
        cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)           # Green: actual contour
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2) # Blue: bounding box
        cv2.circle(img, (int(cx), int(cy)), int(radius), (0, 0, 255), 2)  # Red: enclosing circle

cv2.imshow('Contour Analysis', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
