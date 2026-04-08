# Concept: Convolution Kernels & Padding
# A kernel is a small matrix that slides over an image and transforms each pixel
# based on its neighbors. This is how sharpening, blurring, and edge detection work.
# Padding adds a border around the image — useful when processing near the edges.

import cv2
import numpy as np
from constants import input_img

img = cv2.imread(input_img)

# --- CUSTOM KERNELS ---
# These are hand-crafted kernels that do specific transformations.
# The center value is the weight on the current pixel; surrounding values affect neighbors.

# Sharpen: high center weight, negative neighbors — amplifies contrast at edges
sharpen = np.array([[0, -1, 0],
                    [-1, 10, -1],
                    [0, -1, 0]])

# Emboss: creates a raised/3D-relief effect using an asymmetric gradient kernel
emboss  = np.array([[-2, -1, 0],
                    [-1,  1, 1],
                    [ 0,  1, 2]])

# filter2D applies the kernel: for each pixel, multiply neighbors by kernel, sum them up
# -1 means the output has the same depth (bit depth) as the input
res_sharp  = cv2.filter2D(img, -1, sharpen)
res_emboss = cv2.filter2D(img, -1, emboss)

# --- PADDING (BORDERS) ---
# Padding adds pixels around the image. Useful before filtering so edge pixels aren't ignored.

# BORDER_CONSTANT: fills border with a solid color (red here)
constant = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[0, 0, 255])

# BORDER_REFLECT: mirrors the image at the edge — smoother than a solid color
reflect  = cv2.copyMakeBorder(img, 150, 150, 150, 150, cv2.BORDER_REFLECT)

cv2.imshow('Sharpened', res_sharp)
cv2.imshow('Embossed', res_emboss)
cv2.imshow('Constant Border (red)', constant)
cv2.imshow('Reflected Border', reflect)
cv2.waitKey(0)
cv2.destroyAllWindows()
