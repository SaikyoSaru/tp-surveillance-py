import cv2
import numpy as np

YOLO_CFG = 'src/yolo/cfg/yolov3.cfg'
YOLO_W = 'src/yolo/weights/yolov3.weights'
COCO_NAMES = 'src/yolo/data/coco.names'

class YoloObjectClassifier:
	def __init__(self, confidence_limit=0.5, threshold=0.3):
		self.threshold = threshold
		self.confidence_limit = confidence_limit
		self.net = cv2.dnn.readNet(YOLO_W, YOLO_CFG)
		self.classes = []
		with open(COCO_NAMES, 'r') as coco:
			self.classes = [line.strip() for line in coco.readlines()]
		self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
		layers_names = self.net.getLayerNames()
		self.output_layers = [layers_names[i[0]-1]
					for i in self.net.getUnconnectedOutLayers()]
		
	def get_box_dims(self, outputs, height, width):
		# initialize our lists of detected bounding boxes, confidences, and
		# class IDs, respectively
		boxes = []
		confs = []
		class_ids = []
		for output in outputs:
			for detect in output:
				scores = detect[5:]
				class_id = np.argmax(scores)
				conf = scores[class_id]

				if conf > self.confidence_limit:

					center_x, center_y = int(detect[0] * width), int(detect[1] * height)

					w = int(detect[2] * width)
					h = int(detect[3] * height)

					x = int(center_x - w / 2)
					y = int(center_y - h / 2)
					# update our list of bounding box coordinates, confidences,
					# and class IDs
					boxes.append([x, y, int(w), int(h)])
					confs.append(float(conf))
					class_ids.append(class_id)
		return boxes, confs, class_ids


	def draw_labels(self, boxes, confs, class_ids, image):
		idxs = cv2.dnn.NMSBoxes(boxes, confs, self.confidence_limit,
							self.threshold)
		# ensure at least one detection exists
		if len(idxs) > 0:
			# loop over the indexes we are keeping
			for i in idxs.flatten():
				# extract the bounding box coordinates
				x, y, w, h = boxes[i]
				# draw a bounding box rectangle and label on the image
				color = [int(c) for c in self.colors[class_ids[i]]]
				label = str(self.classes[class_ids[i]])
				cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
				text = "{}: {:.4f}".format(label, confs[i])
				cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
								0.5, color, 2)
		else: 
			return image, False
		return image, True


	def detect_objects(self, image):
		h, w, c = image.shape

		blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
								swapRB=True, crop=False)

		self.net.setInput(blob)
		layer_outputs = self.net.forward(self.output_layers)

		boxes, confs, class_ids = self.get_box_dims(layer_outputs, h, w)

		image, object_identified = self.draw_labels(boxes, confs, class_ids, image)

		return image, object_identified
