import math
import cv2
import numpy as np

def filter(img, x, y, kernel):
	magnitude = 0.
	for i in xrange(len(kernel)):
		for j in xrange(len(kernel[0])):
			xn = x + i - 1
			yn = y + j - 1
			magnitude += img[xn][yn] * kernel[i][j]
	return magnitude

def MyConvolve(img, ff):
	result = np.zeros(img.shape)
	for i in xrange(1, len(result)-1):
		for j in xrange(1, len(result[0])-1):
			magnitude_x = filter(img, i, j, ff[0])
			magnitude_y = filter(img, i, j, ff[1])
			result[i][j] = math.sqrt(magnitude_x**2 + magnitude_y**2)
	return result

def edge_thinning(img):
	for i in xrange(1, len(img)-1):
		for j in xrange(1, len(img[0])-1):
			if img[i][j] > img[i-1][j] and img[i][j] > img[i+1][j]:
				continue
			if img[i][j] > img[i][j-1] and img[i][j] > img[i][j+1]:
				continue
			img[i][j] = 0
	return img

SOBEL_X = [[-1,0,1], 
		   [-2,0,2],
	       [-1,0,1]]

SOBEL_Y = [[1,2,1], 
		   [0,0,0],
	       [-1,-2,-1]]

PREWIT_X = [[-1,0,1], 
		   [-1,0,1],
	       [-1,0,1]]

PREWIT_Y = [[1,1,1], 
		   [0,0,0],
	       [-1,-1,-1]]

IMG_NAMES = ['test1.jpg', 'test2.jpg', 'test3.jpg']

for name in IMG_NAMES:
	img = cv2.imread(name, 0)
	sobel = MyConvolve(img, [SOBEL_X, SOBEL_Y])
	cv2.imwrite('sobel_' + name, sobel)

	thinned_sobel = edge_thinning(sobel)
	cv2.imwrite('thinned_sobel_' + name, thinned_sobel)

	prewit = MyConvolve(img, [PREWIT_X, PREWIT_Y])
	cv2.imwrite('prewit_' + name, prewit)

	thinned_prewit = edge_thinning(prewit)
	cv2.imwrite('thinned_prewit_' + name, thinned_prewit)
