import cv2

img = cv2.imread('input.png')
print(img.shape)
img = cv2.resize(img, (500, 500))
print(img.shape)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# show the images and their pixel values

cv2.imshow('Original', img)
print("Original", img[0][0])

cv2.imshow('Gray', gray)
print("Gray", gray[0][0])

cv2.imshow('HSV', hsv)
print("HSV", hsv[0][0])

cv2.imshow('LAB', lab)
print("LAB", lab[0][0])

cv2.imshow('RGB', rgb)
print("RGB", rgb[0][0])

cv2.waitKey(0)
cv2.destroyAllWindows()
