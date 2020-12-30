from imutils.video import VideoStream
import datetime
import cv2
import imutils
import numpy as np
import time
from threading import Thread, Lock
from object_identifier import YoloObjectClassifier

FACECASCADEPATH = 'src/haarcascades/haarcascade_frontalface_default.xml'
BODYCASCADEPATH = 'src/haarcascades/haarcascade_upperbody.xml'

YOLO_CFG = 'src/yolo/cfg/yolov3.cfg'
YOLO_W = 'src/yolo/weights/yolov3.weights'
COCO_NAMES = 'src/yolo/data/coco.names'
confidence_limit = 0.5
threshold = 0.3


class Detector(Thread):
	def __init__(self, queue, accumWeight=0.1, minimal_frame_count=1):
		Thread.__init__(self)
		self.MINIMAL_AREA = 500
		self.motion_status = False
		self.accumWeight = accumWeight
		self.vs = None
		self.output_frame = None
		self.lock = Lock()
		self.minimal_frame_count = minimal_frame_count
		self.background_model = None
		self.queue = queue
		self.yolo_classifier = YoloObjectClassifier()
		self.pause_motion = False

	def update_background(self, frame):
		# if the background model is None, initialize it
		if self.background_model is None:
			self.background_model = frame.copy().astype("float")
			return
		# update the background model by accumulating the weighted
		# average
		cv2.accumulateWeighted(frame, self.background_model, self.accumWeight)


	def detect_motion(self, frame):
		frameDelta = cv2.absdiff(self.background_model.astype('uint8'), frame)
		thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
		# dilate the thresholded image to fill in holes, then find contours
		# on thresholded image
		thresh = cv2.erode(thresh, None, iterations=2)
		thresh = cv2.dilate(thresh, None, iterations=2)
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
						cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		# loop over the contours
		(minX, minY) = (np.inf, np.inf)
		(maxX, maxY) = (-np.inf, -np.inf)
		self.motion_status = False
		if len(cnts) == 0:
			return None
			
		for c in cnts:
			# if the contour is too small, ignore it
			(x, y, w, h) = cv2.boundingRect(c)
			(minX, minY) = (min(minX, x), min(minY, y))
			(maxX, maxY) = (max(maxX, x + w), max(maxY, y + h))
			self.motion_status = True

		return (thresh, (minX, minY, maxX, maxY))

	def preprocess_image(self, frame):
		# remove color
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# blur image / remove shit
		gray = cv2.GaussianBlur(gray, (7, 7), 0)
		timestamp = datetime.datetime.now()
		cv2.putText(frame, timestamp.strftime(
				"%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

		return gray

	def run(self, video_path=None):
		# video stream from the camera
		
		if video_path is None:
			vs = VideoStream(src=0).start()
			time.sleep(2.0)
		else:
			vs = cv2.VideoCapture(video_path)

		firstFrame = None
		total = 0

		while True:
			motion = None
			frame = vs.read()
		
			frame = frame if video_path is None else frame[1]
			# resize frame
			frame = imutils.resize(frame, width=500)
			gray = self.preprocess_image(frame)

			if total >= self.minimal_frame_count and not self.motion_status:
				motion = self.detect_motion(gray)

			if self.motion_status:
				frame, object_identified = self.yolo_classifier.detect_objects(frame)
				if not object_identified:
					motion = self.detect_motion(gray)
					
			if motion is not None:
				(thresh, (minX, minY, maxX, maxY)) = motion
				cv2.rectangle(frame, (minX, minY), (maxX, maxY),
								(0, 0, 255), 2)

			self.update_background(gray)
			
			total += 1
			self.queue.put(frame.copy())
			

	def create_image(self):
		while True:
			with self.lock:
				if self.output_frame is None:
					continue
					# encode the frame in JPEG format
				(flag, encodedImage) = cv2.imencode(".jpg", self.outputFrame)
				# ensure the frame was successfully encoded
				if not flag:
					continue
			# yield the output frame in the byte format
			yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
						bytearray(encodedImage) + b'\r\n')

def check_for_motion():
	return

def detect_object(frame):
	print("motion")
	return


def detect_human(frame):
	"""
	Function for detecting if there exists any human face in the image
	"""
	body_cascade = cv2.CascadeClassifier(BODYCASCADEPATH)
	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = body_cascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE
	)

	if len(faces):
		for (x, y, w, h) in faces:
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	return frame, len(faces)





def detect_faces(image_path):
	face_cascade = cv2.CascadeClassifier(FACECASCADEPATH)
	img = cv2.imread(image_path)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(
		gray, 
		scaleFactor = 1.1,
		minNeighbors = 5,
		minSize = (30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE
	)

	print(f"Found {len(faces)} faces")
	for (x, y, w, h) in faces:
		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
	cv2.imshow("Faces found", img)
	cv2.waitKey(0)
	return


if __name__ == "__main__":
	detector = Detector()
	detector.run()
	
	# detect_objects(image_path)
	# detect_faces(image_path)
	# detect_human(image_path)
