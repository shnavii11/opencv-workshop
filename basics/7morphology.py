# Concept: Morphological Operations
# Morphology works on binary images (black and white) by growing or shrinking white regions.
# Think of it as sculpting: erosion chisels away the edges, dilation puffs them out.
# Combining these gives you powerful noise removal and shape cleanup tools.

import cv2
import numpy as np
from constants import erosion_dilation_img, hat_input_img, opening_input_img, closing_input_img

# --- EROSION & DILATION ---
img = cv2.imread(erosion_dilation_img, 0)              # Load as grayscale (single channel)
_, threshed = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)  # Convert to pure B&W

# The kernel defines the "neighborhood" shape — a 3x3 square here
kernel = np.ones((3, 3), np.uint8)

eroded  = cv2.erode(threshed, kernel, iterations=5)    # Shrinks white areas (removes noise at edges)
dilated = cv2.dilate(threshed, kernel, iterations=5)   # Grows white areas (fills small holes)

# Morphological Gradient = Dilation - Erosion = just the outline of white regions
grad = cv2.morphologyEx(threshed, cv2.MORPH_GRADIENT, kernel)

cv2.imshow('Original', img)
cv2.imshow('Thresholded (cleaner view)', threshed)
cv2.imshow('Erode', eroded)
cv2.imshow('Dilate', dilated)
cv2.imshow('Gradient (outline)', grad)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- OPENING & CLOSING ---
# Opening = Erode then Dilate: removes small white noise dots
to_open = cv2.imread(opening_input_img)
to_open = cv2.cvtColor(to_open, cv2.COLOR_BGR2GRAY)

# Closing = Dilate then Erode: fills small black holes inside white regions
to_close = cv2.imread(closing_input_img)
to_close = cv2.cvtColor(to_close, cv2.COLOR_BGR2GRAY)

opening = cv2.morphologyEx(to_open,  cv2.MORPH_OPEN,  kernel)
closing = cv2.morphologyEx(to_close, cv2.MORPH_CLOSE, kernel)

cv2.imshow('Opening Input', to_open)
cv2.imshow('Opening Result (noise removed)', opening)
cv2.imshow('Closing Input', to_close)
cv2.imshow('Closing Result (holes filled)', closing)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- TOP HAT & BLACK HAT ---
# These reveal fine structures that would otherwise be invisible
img = cv2.imread(hat_input_img)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel = np.ones((10, 10), np.uint8)   # Larger kernel to capture bigger structures

# Top Hat = Original - Opening: highlights bright details smaller than the kernel
tophat   = cv2.morphologyEx(gray_img, cv2.MORPH_TOPHAT,   kernel)

# Black Hat = Closing - Original: highlights dark details smaller than the kernel
blackhat = cv2.morphologyEx(gray_img, cv2.MORPH_BLACKHAT, kernel)

cv2.imshow('Top Hat (bright details)', tophat)
cv2.imshow('Black Hat (dark details)', blackhat)
cv2.waitKey(0)
cv2.destroyAllWindows()
