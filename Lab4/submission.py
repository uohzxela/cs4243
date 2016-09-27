import cv2
import numpy as np

def get_magnitude(img, x, y, kernel):
	magnitude = 0.
	for i in xrange(len(kernel)):
		for j in xrange(len(kernel[0])):
			xn = x + i - 1
			yn = y + j - 1
			magnitude += img[xn][yn] * kernel[i][j]
	return magnitude

def flip(f):
	# flip left right
	for i in xrange(len(f)):
		n = len(f[i])
		for j in xrange(n/2):
			f[i][j], f[i][n-j-1] = f[i][n-j-1], f[i][j]
	# flip up down
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
			result[i][j] = get_magnitude(img, i, j, kernel)
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

def compute_response_matrix(W_xx, W_xy, W_yy, step_size):
	response_matrix = np.zeros((W_xx.shape[0]/step_size, W_xx.shape[1]/step_size))
	for i in xrange(step_size, len(W_xx), step_size):
		for j in xrange(step_size, len(W_xx[0]), step_size):
			W = np.array([[W_xx[i][j], W_xy[i][j]], [W_xy[i][j], W_yy[i][j]]])
			detW = np.linalg.det(W)
			traceW = np.trace(W)
			response = detW - (0.06 * traceW * traceW)
			response_matrix[(i/step_size)-1][(j/step_size)-1] = response
	return response_matrix

def compute_corners(W_xx, W_xy, W_yy, step_size=10, threshold=0.1):
	response_matrix = compute_response_matrix(W_xx, W_xy, W_yy, step_size)
	max_response = response_matrix.max()
	corners = []
	for i in xrange(len(response_matrix)):
		for j in xrange(len(response_matrix[0])):
			response = response_matrix[i][j]
			if response >= threshold * max_response:
				corners.append(((i+1)*step_size, (j+1)*step_size))
	return corners

def draw_corners_and_save_image(orig, name, corners):
	for x, y in corners:
		cv2.rectangle(orig, (y, x), (y+9, x+9), (0,255,0), 2)
	cv2.imwrite(name, orig)

SOBEL_X = [[-1,0,1], 
		   [-2,0,2],
	       [-1,0,1]]

SOBEL_Y = [[1,2,1], 
		   [0,0,0],
	       [-1,-2,-1]]

GAUSSIAN = get_gauss_kernel(size=3)

IMG_NAMES = ['checker.jpg', 'flower.jpg', 'test1.jpg', 'test2.jpg', 'test3.jpg']

for name in IMG_NAMES:
	img = cv2.imread(name, 0)
	orig = cv2.imread(name)
	orig2 = orig.copy()

	gx = convolve(img, SOBEL_X)
	gy = convolve(img, SOBEL_Y)

	I_xx = gx * gx
	I_xy = gx * gy
	I_yy = gy * gy

	W_xx = convolve(I_xx, GAUSSIAN)
	W_xy = convolve(I_xy, GAUSSIAN)
	W_yy = convolve(I_yy, GAUSSIAN)

	
	corners = compute_corners(W_xx, W_xy, W_yy, step_size=10, threshold=0.1)
	more_corners = compute_corners(W_xx, W_xy, W_yy, step_size=1, threshold=0.1)

	draw_corners_and_save_image(orig2, 
								name.split(".")[0] + "_corners.jpg",
								corners)
	draw_corners_and_save_image(orig, 
								name.split(".")[0] + "_more_corners.jpg",
								more_corners)
