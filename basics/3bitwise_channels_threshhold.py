import cv2
import numpy as np

img = cv2.imread('input.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# BITWISE OPERATIONS

# bitwise_not: Inverts the mask (White becomes Black, Black becomes White)
inverted_img = cv2.bitwise_not(img)
cv2.imshow('Original', img)
cv2.imshow('NOT', inverted_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

# mask  is same size as image in grayscale
mask = np.zeros(img.shape[:2], dtype="uint8")
# draw a white circle on our black mask
# white area (255) is what we want to keep of the original image
cv2.circle(mask, (250, 250), 200, 255, -1)

# BITWISE OPERATIONS

mask_inv = cv2.bitwise_not(mask)
# bitwise_and: Keeps only the pixels where the mask is white
masked_img = cv2.bitwise_and(img, img, mask=mask)

# bitwise_or: Using the inverted mask to "brighten" or merge (advanced demo)
masked_img_or = cv2.bitwise_or(img, img, mask=mask)
# Usually, we use 'and' for masking, but 'not' is great for showing background subtraction

# 4. DISPLAY
cv2.imshow("Original", img)
cv2.imshow("The Mask", mask)
cv2.imshow("Mask NOT", mask_inv)
cv2.imshow("Masked Output (The Cutout)", masked_img)
cv2.imshow("Masked image using OR", masked_img_or)

cv2.waitKey(0)
cv2.destroyAllWindows()


# Geometry
cropped = img[100:400, 100:400] # [y_start:y_end, x_start:x_end]
flipped = cv2.flip(img, 1) # 1=Horizontal, 0=Vertical

# Thresholding
# Simple Binary
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# Adaptive (Handles shadows better)
adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 11, 2)

cv2.imshow('Cropped', cropped)
cv2.imshow('Flipped', flipped)
cv2.imshow('Binary', thresh)
cv2.imshow('Adaptive', adaptive)
cv2.waitKey(0)
cv2.destroyAllWindows()