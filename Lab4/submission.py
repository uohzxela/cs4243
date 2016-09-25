import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

"""
need to flip?
within 10%?
"""
def filter(img, x, y, kernel):
	magnitude = 0.
	for i in xrange(len(kernel)):
		for j in xrange(len(kernel[0])):
			xn = x + i - 1
			yn = y + j - 1
			magnitude += img[xn][yn] * kernel[i][j]
	return magnitude

def flip(ff):
	f_x, f_y = ff
	# flip horizontally for filter x
	for i in xrange(len(f_x)):
		n = len(f_x[i])
		for j in xrange(n/2):
			f_x[i][j], f_x[i][n-j-1] = f_x[i][n-j-1], f_x[i][j]

	# flip vertically for filter y
	for i in xrange(len(f_y[0])):
		n = len(f_y)
		for j in xrange(n/2):
			f_y[j][i], f_y[n-j-1][i] = f_y[n-j-1][i], f_y[j][i]

	return ff

def convolve(img, kernel):
	result = np.zeros(img.shape)
	for i in xrange(1, len(result)-1):
		for j in xrange(1, len(result[0])-1):
			result[i][j] = filter(img, i, j, kernel)
	return result

def get_gauss_kernel(size,sigma=1.0):
	## returns a 2d gaussian kernel
	if size<3:
		size = 3
	m = size/2
	x, y = np.mgrid[-m:m+1, -m:m+1]
	kernel = np.exp(-(x*x + y*y)/(2*sigma*sigma)) 
	kernel_sum = kernel.sum()
	if not sum==0:
		kernel = kernel/kernel_sum 
	return kernel

def compute_response_matrix(W_xx, W_xy, W_yy):
	response_matrix = np.zeros((W_xx.shape[0]/10, W_xx.shape[1]/10))
	for i in xrange(10, len(W_xx), 10):
		for j in xrange(10, len(W_xx[0]), 10):
			W = np.array([[W_xx[i][j], W_xy[i][j]], [W_xy[i][j], W_yy[i][j]]])
			detW = np.linalg.det(W)
			traceW = np.trace(W)
			response_matrix[(i/10)-1][(j/10)-1] = detW - (0.06 * traceW * traceW)
	return response_matrix

def find_corners(response_matrix):
	max_response = response_matrix.max()
	corners = []
	for i in xrange(len(response_matrix)):
		for j in xrange(len(response_matrix[0])):
			response = response_matrix[i][j]
			if response >= 0.9 * max_response:
				corners.append(((i+1)*10, (j+1)*10))
	return corners

SOBEL_X = [[-1,0,1], 
		   [-2,0,2],
	       [-1,0,1]]

SOBEL_Y = [[1,2,1], 
		   [0,0,0],
	       [-1,-2,-1]]

IMG_NAMES = ['flower.jpg', 'checker.jpg']

for name in IMG_NAMES:
	img = cv2.imread(name, 0)
	gx = convolve(img, SOBEL_X)
	gy = convolve(img, SOBEL_Y)

	I_xx = gx * gx
	I_xy = gx * gy
	I_yy = gy * gy

	gauss_kernel = get_gauss_kernel(size=3)
	print gauss_kernel
	W_xx = convolve(I_xx, gauss_kernel)
	W_xy = convolve(I_xy, gauss_kernel)
	W_yy = convolve(I_yy, gauss_kernel)

	response_matrix = compute_response_matrix(W_xx, W_xy, W_yy)
	corners = find_corners(response_matrix)
	rows, cols = zip(*corners)
	# print corners
	plt.figure()
	plt.imshow(img, cmap='gray')
	plt.hold(True)
	plt.scatter(rows,cols,color='blue')
	plt.show()