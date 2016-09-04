import cv2
import math
import numpy as np 

def get_histogram(img):
	m, n = len(img), len(img[0])
	hist = [0.] * 256
	for i in xrange(m):
		for j in xrange(n):
			hist[img[i, j, 0]] += 1
	return np.array(hist)/(m*n)

def histeq(img):
	hist = get_histogram(img)
	cum = np.array([sum(hist[:i+1]) for i in xrange(len(hist))])
	cum = np.uint8(255 * cum)
	img_eq = np.zeros_like(img)
	for i in xrange(len(img)):
		for j in xrange(len(img[0])):
			img_eq[i, j] = cum[img[i, j]]
	return img_eq

def hsv_to_bgr(img):
	for i in xrange(len(img)):
		for j in xrange(len(img[0])):
			h, s, v = img[i][j]
			# denormalize into actual hsv values
			h *= 2.
			s /= 255.
			v /= 255.
			r, g, b = hsv2rgb(h, s, v)
			img[i][j][0] = b
			img[i][j][1] = g
			img[i][j][2] = r
	return img

def bgr_to_hsv(img):
	for i in xrange(len(img)):
		for j in xrange(len(img[0])):
			b, g, r = img[i][j]
			h, s, v = rgb2hsv(r, g, b)
			# normalizing hsv to [0,255] range
			h /= 2
			s *= 255
			v *= 255
			img[i][j][0] = h
			img[i][j][1] = s
			img[i][j][2] = v
	return img

def hsv2rgb(h, s, v):
    h, s, v = float(h), float(s), float(v)
    c = v * s
    x = c * (1 - abs(((h / 60) % 2) - 1))
    m = v - c
    if 0 <= h < 60:
    	r, g, b = c, x, 0
    elif 60 <= h < 120:
    	r, g, b = x, c, 0
    elif 120 <= h < 180:
    	r, g, b = 0, c, x
    elif 180 <= h < 240:
    	r, g, b = 0, x, c
    elif 240 <= h < 300:
    	r, g, b = x, 0, c
    else:
    	r, g, b = c, 0, x
    r, g, b = (r+m)*255, (g+m)*255, (b+m)*255
    return r, g, b

def rgb2hsv(r, g, b):
    r, g, b = r/255., g/255., b/255.
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    diff = cmax - cmin
    if diff == 0:
        h = 0
    elif cmax == r:
        h = 60 * (((g-b)/diff) % 6)
    elif cmax == g:
        h = 60 * (((b-r)/diff) + 2)
    elif cmax == b:
        h = 60 * (((r-g)/diff) + 4)
    if cmax == 0:
        s = 0
    else:
        s = diff/cmax
    v = cmax
    return h, s, v

def get_hsv(name):
	h = cv2.imread(name + "_hue." + FILE_TYPE)
	s = cv2.imread(name + "_saturation." + FILE_TYPE)
	v = cv2.imread(name + "_brightness." + FILE_TYPE)
	return h, s, v

def combine_hsv(h, s, v):
	h[:,:,1] = s[:,:,1]
	h[:,:,2] = v[:,:,2]
	return h

IMG_DIR = "lab2_pictures/"
IMG_NAMES = ["concert", "sea1", "sea2"]
FILE_TYPE = 'jpg'

"""
1) Perform RGB to HSV conversions on the following images and store the output images.
"""
for name in IMG_NAMES:
	img = cv2.imread(IMG_DIR + name + "." + FILE_TYPE)
	img = bgr_to_hsv(img)
	cv2.imwrite(name + "_hue." + FILE_TYPE, img[:,:,0])
	cv2.imwrite(name + "_saturation." + FILE_TYPE, img[:,:,1])
	cv2.imwrite(name + "_brightness." + FILE_TYPE, img[:,:,2])

"""
2) Perform HSV to RGB conversions on the HSV images obtained in Step 1 above,
and store the output images.
"""
for name in IMG_NAMES:
	h, s, v = get_hsv(name)
	img = hsv_to_bgr(combine_hsv(h, s, v))
	cv2.imwrite(name + "_hsv2rgb." + FILE_TYPE, img)

"""
3) Perform histogram equalizations on the value channels of the HSV images obtained in Step 1. 
Combine this histogram equalized value channel with the original hue and saturation, 
then convert the HSV images to RGB and save these RGB images.
"""
for name in IMG_NAMES:
	h, s, v = get_hsv(name)
	v = histeq(v)
	img = hsv_to_bgr(combine_hsv(h, s, v))
	cv2.imwrite(name + "_histeq." + FILE_TYPE, img)
