from functools import lru_cache
from itertools import product
from math import inf
import cv2
import numpy as np
import sys

sys.setrecursionlimit(100000)

img = cv2.imread('Broadway_tower_edit.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Sobel
img_sobelx = cv2.Sobel(img_gray, cv2.CV_8U, 1, 0, ksize=3)


cv2.imshow('Original', img)
# cv2.imshow('Sobel', img_sobelx)#
cv2.waitKey(0)

cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
cv2.namedWindow('Line to carve', cv2.WINDOW_NORMAL)
cv2.namedWindow('Carved', cv2.WINDOW_NORMAL)

carved_indices = set()

for it in range((img.shape[1])//2):
    height, width, _ = img.shape

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#.astype(float)

    gX = cv2.Sobel(img_gray, cv2.CV_8U, 1, 0, ksize=3)#.astype(np.float32)
    gY = cv2.Sobel(img_gray, cv2.CV_8U, 0, 1, ksize=3)#.astype(np.float32)

    img_magnitude = np.sqrt((gX.astype(float) ** 2) + (gY.astype(float) ** 2)) ** 2
    cv2.imshow("gX", gX)
    cv2.imshow("gY", gY)
    img_magnitude_normalized = img_magnitude / np.max(img_magnitude)
    img_magnitude_normalized = (img_magnitude_normalized * 255).astype(np.uint8)
    cv2.imshow("magnitude", img_magnitude_normalized)

    height, width = img_magnitude.shape

    def mag(i, j, current):
        if i < 0 or j < 0 or i >= height or j >= width:
            return current
        return abs(img_magnitude[i, j])
    def mag_neighbour(i, j):
        current = abs(img_magnitude[i, j])
        res = 0
        for di, dj in product(range(-3, 3), range(-3, 3)):
            # print(di, dj)
            res += mag(i + di, j + dj, current)
        return res
    
    @lru_cache(maxsize=None)
    def dp(i, j):
        if i < 0:
            return inf
        if j < 0:
            return inf
        if j >= width:
            return inf
        if i >= height:
            return inf
        
        if  i == height - 1:
            return mag_neighbour(i, j)
        return mag_neighbour(i, j) + min(dp(i + 1, j - 1), dp(i + 1, j), dp(i + 1, j + 1))
    min_index = 0
    min_val = inf
    
    for j in range(len(img[1])):
        val = dp(0, j)
        if val < min_val:
            min_val = val
            min_index = j

    img2 = img.copy()
    carved_indices = set()
    for i in range(height):
        carved_indices.add((i, min_index))
        img2[i, min_index] = [255, 0, 255]
        min_index = min_index + np.argmin([dp(i + 1, min_index - 1), dp(i + 1, min_index), dp(i + 1, min_index + 1)]) - 1

    cv2.imshow('Line to carve', img2)
    for i, j in carved_indices:
        img[i, j:-1] = img[i, j + 1:]
    img = img[:, :-1]

    cv2.imshow('Carved', img)
    key = cv2.waitKey() & 0xFF
    if key == ord('q'):
        break
    