import cv2
from constants import input_img
# Read image from the given file path
img = cv2.imread(input_img)

# Check if image is loaded properly
if img is None:
    print("Image not found!")
    exit()

# Display original image
cv2.imshow('Original Image', img)

# Resize image to fixed dimensions (300x200)
resized_image = cv2.resize(img, (300, 200))
cv2.imshow('Resized (300x200)', resized_image)

# Resize image to half of its original size
h, w = img.shape[:2]
half = cv2.resize(img, (w // 2, h // 2))
cv2.imshow('Half Size', half)

# Resize using interpolation
resized_inter = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)
cv2.imshow('Resized with Interpolation', resized_inter)

# Crop a specific region
crop = img[50:200, 10:400]
cv2.imshow('Cropped Region', crop)

# Crop top-left corner
corner = img[0:200, 0:200]
cv2.imshow('Top-Left Corner', corner)

# Center crop
cy, cx = h // 2, w // 2
size = 150
center = img[cy-size:cy+size, cx-size:cx+size]
cv2.imshow('Center Crop', center)

# Flip horizontally
flip_h = cv2.flip(img, 1)
cv2.imshow('Horizontal Flip', flip_h)

# Flip vertically
flip_v = cv2.flip(img, 0)
cv2.imshow('Vertical Flip', flip_v)

# Flip both axes
flip_both = cv2.flip(img, -1)
cv2.imshow('Flip Both Axes', flip_both)

# Save image (format detected from extension)
cv2.imwrite('assets/output.png', img)

# Save JPEG with quality (0 = lowest, 100 = best)
cv2.imwrite('assets/output.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])

# Save PNG with compression (0 = none, 9 = max)
cv2.imwrite('assets/output_compressed.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 3])

# Wait until key press
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()