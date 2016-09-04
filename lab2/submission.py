import cv2
import math
import pdb

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
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b
    
def rgb2hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx
    return h, s, v

def histogram_eq(v):
	pass

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
# for name in IMG_NAMES:
# 	h, s, v = get_hsv(name)
# 	v = histogram_eq(v)
# 	img = hsv_to_bgr(combine_hsv(h, s, v))
# 	cv2.imwrite(name + "_histeq." + FILE_TYPE, img)
