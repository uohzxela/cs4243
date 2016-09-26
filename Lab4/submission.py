import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

def filter(img, x, y, kernel):
	magnitude = 0.
	for i in xrange(len(kernel)):
		for j in xrange(len(kernel[0])):
			xn = x + i - 1
			yn = y + j - 1
			magnitude += img[xn][yn] * kernel[i][j]
	return magnitude

def flip(f):
	# flip horizontally
	for i in xrange(len(f)):
		n = len(f[i])
		for j in xrange(n/2):
			f[i][j], f[i][n-j-1] = f[i][n-j-1], f[i][j]
	# flip vertically
	for i in xrange(len(f)):
		n = len(f)
		for j in xrange(n/2):
			f[j][i], f[n-j-1][i] = f[n-j-1][i], f[j][i]
	return f

def convolve(img, kernel):
	kernel = flip(kernel)
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

def compute_responses(W_xx, W_xy, W_yy):
	responses = []
	max_response = float('-inf')
	STEP = 10
	for i in xrange(STEP, len(W_xx), STEP):
		for j in xrange(STEP, len(W_xx[0]), STEP):
			W = np.array([[W_xx[i][j], W_xy[i][j]], [W_xy[i][j], W_yy[i][j]]])
			detW = np.linalg.det(W)
			traceW = np.trace(W)
			response = detW - (0.06 * traceW * traceW)
			max_response = max(response, max_response)
			responses.append((response, i, j))
	return responses, max_response

def find_corners(responses, max_response):
	# max_response = response_matrix.max()
	corners = []
	for i in xrange(len(responses)):
		response, x, y = responses[i]
		if response >= 0.9 * max_response:
			corners.append((x, y))
	return corners

SOBEL_X = [[-1,0,1], 
		   [-2,0,2],
	       [-1,0,1]]

SOBEL_Y = [[1,2,1], 
		   [0,0,0],
	       [-1,-2,-1]]

IMG_NAMES = ['checker.jpg', 'flower.jpg', 'test1.jpg', 'test2.jpg', 'test3.jpg']

for name in IMG_NAMES:
	img = cv2.imread(name, 0)
	orig = cv2.imread(name)
	gx = convolve(img, SOBEL_X)
	gy = convolve(img, SOBEL_Y)

	I_xx = gx * gx
	I_xy = gx * gy
	I_yy = gy * gy

	gauss_kernel = get_gauss_kernel(size=3)
	W_xx = convolve(I_xx, gauss_kernel)
	W_xy = convolve(I_xy, gauss_kernel)
	W_yy = convolve(I_yy, gauss_kernel)

	responses, max_response = compute_responses(W_xx, W_xy, W_yy)
	corners = find_corners(responses, max_response)
	for x, y in corners:
		cv2.circle(orig, (x, y), 9, (0,255,0), 2)
	cv2.imwrite(name.split(".")[0] + "_corners.jpg", orig)
	# rows, cols = zip(*corners)
	# plt.figure()
	# plt.imshow(img, cmap='gray')
	# plt.hold(True)
	# plt.scatter(rows,cols,color='blue')
	# plt.show()