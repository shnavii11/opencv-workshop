# Concept: Bitwise Operations & Thresholding
# Bitwise ops treat each pixel like a binary number and apply logic (NOT, AND, OR).
# Thresholding converts a grayscale image into pure black-and-white based on a cutoff value.

import cv2
import numpy as np
from constants import input_img

img = cv2.imread(input_img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- BITWISE NOT ---
# Inverts every pixel: 0 becomes 255, 255 becomes 0 (like a photo negative)
inverted_img = cv2.bitwise_not(img)
cv2.imshow('Original', img)
cv2.imshow('NOT (Inverted)', inverted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- MASKING WITH BITWISE AND ---
# A mask is a grayscale image where white (255) = "keep" and black (0) = "discard"
mask = np.zeros(img.shape[:2], dtype="uint8")   # Start with a fully black (discard all) mask
cv2.circle(mask, (250, 250), 200, 255, -1)       # Draw a white circle — only this area will show
mask=cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # Convert mask to 3 channels to match img for bitwise ops
mask_inv = cv2.bitwise_not(mask)                 # Invert the mask (now everything OUTSIDE shows)

# bitwise_and keeps only the pixels where the mask is white (255)
masked_img = cv2.bitwise_and(img,mask,mask=None) #masked_img = cv2.bitwise_and(img, mask)  # This also works since mask is already 3-channel

# bitwise_or is less strict — non-zero pixels from either source come through
masked_img_or = cv2.bitwise_or(img,mask,mask=None) #masked_img_or = cv2.bitwise_or(img, mask)  # This also works since mask is already 3-channel

cv2.imshow("Original", img)
cv2.imshow("The Mask", mask)
cv2.imshow("Mask NOT (Inverted)", mask_inv)
cv2.imshow("Masked Output (bitwise_and)", masked_img)
cv2.imshow("Masked image using OR (bitwise_or)", masked_img_or)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- GEOMETRY ---
cropped = img[100:400, 100:400]     # Slice the array like a 2D list: [y_start:y_end, x_start:x_end]
flipped = cv2.flip(img, 1)          # 1 = horizontal flip, 0 = vertical flip

# --- THRESHOLDING ---
# Simple Binary: if pixel > 127, set to 255 (white); otherwise 0 (black)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Adaptive: uses a local neighborhood average instead of one global value
# Handles uneven lighting much better than simple thresholding
adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)

cv2.imshow('Cropped', cropped)
cv2.imshow('Flipped', flipped)
cv2.imshow('Binary Threshold', thresh)
cv2.imshow('Adaptive Threshold', adaptive)
cv2.waitKey(0)
cv2.destroyAllWindows()
