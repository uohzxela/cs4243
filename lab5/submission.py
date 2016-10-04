import cv2
import cv2.cv as cv
import numpy as np

"""
show background image or video?
why convert to uint8?
"""

cap = cv2.VideoCapture('traffic.mp4')

frame_count = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))
frame_height = int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH))
fps = int(cap.get(cv.CV_CAP_PROP_FPS))

print "Frame count:", frame_count
print "Frame height:", frame_height
print "Frame_width:", frame_width
print "FPS:", fps

_, img = cap.read()
avg_img = np.float32(img)

for fr in range(1, frame_count):
	_, frame = cap.read()
	alpha = 1./float(fr+1)
	cv2.accumulateWeighted(frame, avg_img, alpha)
	norm_img = np.uint8(cv2.convertScaleAbs(avg_img))
	cv2.imshow('normImg', norm_img)

	if fr == frame_count-1:
		cv2.imwrite("background.jpg", norm_img)

	cv2.waitKey(1)

	print "fr = ", fr, " alpha = ", alpha

cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()
