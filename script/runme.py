import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('data/foto-HL-28.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)


cv.namedWindow('image',cv.WINDOW_NORMAL)
cv.imshow("tresh", thresh)
cv.waitKey(0)
cv.destroyAllWindows()
