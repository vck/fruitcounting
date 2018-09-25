import cv2 as cv

img = cv.imread('data/foto-HL-28.jpg', 0)
edges = cv.Canny(img, 350, 260)

(_, hitung, _) = cv.findContours(edges.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

print(len(hitung))
