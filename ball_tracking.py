# USAGE
# python ball_tracking.py --video ball_tracking_example.mp4
# python ball_tracking.py

# import the necessary packages
from collections import deque
from copy import deepcopy
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=10,
	help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
color_range 	= {'green': {'lower': (29, 86, 6), 'upper': (64, 255, 255)},
				  'red':    {'lower': (153, 0, 0), 'upper': (255, 230, 230)}}
tracking_points = {}

for color in color_range:
	tracking_points[color] = deque(maxlen=args["buffer"])

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	camera = cv2.VideoCapture(0)

# otherwise, grab a reference to the video file
else:
	camera = cv2.VideoCapture(args["video"])

# keep looping
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()

	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if args.get("video") and not grabbed:
		break

	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, width=600)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	for color in color_range:
		mask 				   = cv2.inRange(hsv, color_range[color]['lower'], color_range[color]['upper'])
		mask 				   = cv2.erode(mask, None, iterations=2)
		mask 				   = cv2.dilate(mask, None, iterations=2)
		cnts 				   = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
		if len(cnts) > 0:
			c = max(cnts, key=cv2.contourArea)
			((x, y), radius) = cv2.minEnclosingCircle(c)
			M = cv2.moments(c)
			center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
			if radius > 10:
				cv2.circle(frame, (int(x), int(y)), int(radius),
					   	  (0, 255, 255), 2)
				cv2.circle(frame, center, 5, (0, 0, 255), -1)
				tracking_points[color].appendleft(center)
			else:
				tracking_points[color] = deque(maxlen=args["buffer"])
		for i in xrange(1, len(tracking_points[color])):
			if tracking_points[color][i - 1] is None or tracking_points[color][i] is None:
				continue
			thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
			cv2.line(frame, tracking_points[color][i - 1], tracking_points[color][i], (0, 0, 255), thickness)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
