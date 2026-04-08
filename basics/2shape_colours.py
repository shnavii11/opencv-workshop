# Concept: Image Shape & Color Spaces
# An image is just a 3D NumPy array: (height, width, channels).
# The "color space" decides what those channels mean — BGR, grayscale, HSV, etc.

import cv2
from constants import input_img

img = cv2.imread(input_img)
print(img.shape)                        # (height, width, 3) — 3 channels = BGR
img = cv2.resize(img, (500, 500))       # Resize to a fixed 500x500 for consistency
print(img.shape)

# Convert to different color spaces
# OpenCV loads images in BGR by default (not RGB — yes, it's backwards, welcome to OpenCV)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # 1 channel: pure brightness, no color
hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)    # Hue-Saturation-Value: great for color detection
lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)    # Perceptually uniform — used in pro color work
rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # Standard RGB (what most other tools expect)

# Show each version and print the pixel value at position [0][0] (top-left corner)
# Notice how the same pixel looks completely different across color spaces
cv2.imshow('Original', img)
print("Original (BGR)", img[0][0])

cv2.imshow('Gray', gray)
print("Gray", gray[0][0])       # Single value: 0 (black) to 255 (white)

cv2.imshow('HSV', hsv)
print("HSV", hsv[0][0])         # [Hue 0-179, Saturation 0-255, Value 0-255]

cv2.imshow('LAB', lab)
print("LAB", lab[0][0])

cv2.imshow('RGB', rgb)
print("RGB", rgb[0][0])         # Same pixel values as BGR, just R and B swapped

cv2.waitKey(0)
cv2.destroyAllWindows()
