import numpy as np
import cv2
import time
import datetime
import imutils
from imutils.video import VideoStream

MINIMAL_AREA = 500



def detect_motion(video_path=None):
	# video stream from the camera
	if video_path is None:
		vs = VideoStream(src=0).start()
		time.sleep(2.0)
	else:
		vs = cv2.VideoCapture(video_path)
	
	firstFrame = None
	
	while True:
		frame = vs.read()
		
		frame = frame if video_path is None else frame[1]
		
		# cv2.imshow("test", frame)
		
		if frame is None:
			break

		# cv2.imshow("test", frame)
		# resize frame

		frame = imutils.resize(frame, width=500)
		# remove color
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# blur image / remove shit
		gray = cv2.GaussianBlur(gray, (21, 21), 0)

		if firstFrame is None: 	
			firstFrame = gray
			continue
		
		frameDelta = cv2.absdiff(firstFrame, gray)
		thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
		# dilate the thresholded image to fill in holes, then find contours
		# on thresholded image
		thresh = cv2.dilate(thresh, None, iterations=2)
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
								cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		# loop over the contours
		motion_status = "No Motion Detected"
		for c in cnts:
			# if the contour is too small, ignore it
			if cv2.contourArea(c) < MINIMAL_AREA:
				continue
			# compute the bounding box for the contour, draw it on the frame,
			# and update the text
			(x, y, w, h) = cv2.boundingRect(c)
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			motion_status = "Motion Detected"

			# draw the text and timestamp on the frame

		cv2.putText(frame, "Room Status: {}".format(motion_status), (10, 20),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
				(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
		# show the frame and record if the user presses a key
		cv2.imshow("Security Feed", frame)
		# cv2.imshow("Thresh", thresh)
		# cv2.imshow("Frame Delta", frameDelta)
		key = cv2.waitKey(1) & 0xFF
		# if the `q` key is pressed, break from the lop
		if key == ord("q"):
			break


	# cleanup the camera and close any open windows
	vs.stop() if video_path is None else vs.release()
	
	cv2.destroyAllWindows()


if __name__ == "__main__":
	motion_path = 'examples/motion/motion.mov'
	# no_motion = 'examples/motion/minimal_motion.mov'
	detect_motion()
	detect_motion(motion_path)
