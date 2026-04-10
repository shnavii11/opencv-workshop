# Concept: Filtering Techniques in Image Processing
# Filtering is a fundamental technique in image processing used to enhance or modify images.
# Import OpenCV library for computer vision tasks
import cv2

# Import the path to the input image from constants
from constants import filtering_input_img

# Load the image from the specified path
# cv2.imread() reads an image from a file and returns it as a NumPy array
img = cv2.imread(filtering_input_img)

# Apply Gaussian Blur to reduce noise and smooth the image
# GaussianBlur takes the image, kernel size (5x5), and sigma (0 for auto)
gaussian = cv2.GaussianBlur(img, (5, 5), 0)

# Apply Median Blur to remove salt-and-pepper noise
# medianBlur takes the image and kernel size (5)
median = cv2.medianBlur(img, 5)

# Apply Bilateral Filter to smooth while preserving edges
# bilateralFilter takes image, diameter (9), sigma color (75), sigma space (75)
bilateral = cv2.bilateralFilter(img, 9, 75, 75)

# Display the original image in a window titled "Original"
cv2.imshow("Original", img)

# Display the Gaussian blurred image
cv2.imshow("Gaussian", gaussian)

# Display the median blurred image
cv2.imshow("Median", median)

# Display the bilateral filtered image
cv2.imshow("Bilateral", bilateral)

# Wait indefinitely for a key press (0 means wait forever)
# This keeps the windows open until you press any key
cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()