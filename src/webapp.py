from flask import Response
from flask import Flask
from flask import render_template
from threading import Thread, Lock
import datetime
import time
from queue import Queue
import cv2
from detection import Detector

app = Flask(__name__)
img_queue = Queue()
detector = Detector(img_queue)


@app.route('/')
@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/video_feed')
def video_feed():
	return Response(pull_image(), mimetype="multipart/x-mixed-replace; boundary=frame")

def pull_image():
	while True:
		frame = img_queue.get()
		if frame is None:
			continue
			# encode the frame in JPEG format
		(flag, encodedImage) = cv2.imencode(".jpg", frame)
		# ensure the frame was successfully encoded
		if not flag:
			continue
		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +	bytearray(encodedImage) + b'\r\n')


if __name__ == "__main__":
	 
	detector.start()
	app.run(host="localhost", port=8080, threaded=True, use_reloader=False)
