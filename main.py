import cv2 as cv

sample = cv.imread('Sample3.jpg')
cv.imshow("Sample", sample)

gray = cv.cvtColor(sample, cv.COLOR_BGR2GRAY)

threshold, thresh = cv.threshold(gray, 150, 255,cv.THRESH_BINARY)

cv.imshow("Black And White", thresh)
cv.waitKey(0)