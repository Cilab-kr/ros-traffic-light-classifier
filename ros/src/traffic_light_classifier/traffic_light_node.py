#!/usr/bin/env python
import numpy
import sys
import PIL
import rospy

from PIL import Image
from cv_bridge import CvBridge
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from traffic_light_classifier.msg import traffic_light
from sensor_msgs.msg import Image
from std_msgs.msg import Header

RED = 0
GREEN = 1
UNKNOWN = 2
RADIUS_MULTIPLIER = 6

model = None


class BBox(object):
	def __init__(self, x, y, width, height):
		self.x = x
		self.y = y
		self.width = width
		self.height = height


def calculate_bounds(bound):
	xmin = sys.maxint
	xmax = -sys.maxint - 1
	ymin = sys.maxint
	ymax = -sys.maxint - 1

	x = bound.x
	y = bound.y

	xmin = min(xmin, x)
	xmax = max(xmax, x)
	ymin = min(ymin, y)
	ymax = max(ymax, y)

	xmax = xmin + bound.width
	ymax = ymin + bound.height

	return xmin, xmax, ymin, ymax


def crop_image(image, xmin, xmax, ymin, ymax):
	return image.crop((xmin, ymin, xmax, ymax))


def predict_light(cropped_roi):
	# Load CNN Model
	loaded_model = get_model()
	image_array = img_to_array(
		cropped_roi.resize((64, 64), PIL.Image.ANTIALIAS))
	prediction = loaded_model.predict(image_array[None, :])

	if prediction[0][0] == 1:
		return GREEN
	elif prediction[0][1] == 1:
		return RED
	else:
		return UNKNOWN


def get_model():
	global model
	if not model:
		model = load_model('light_classifier_model.h5')
	return model


def detect_signal(image):
	global bounds, detect_cnt
	if bounds.x == 0 and bounds.y == 0:
		# No signals are visible
		traffic_light.recognition_result = UNKNOWN
		light_detected_pub.publish(traffic_light(traffic_light=UNKNOWN))
		return

	if (detect_cnt % 50 != 0):
		detect_cnt += 1
		return

	# Convert the image to PIL
	cv_bridge = CvBridge()
	cv_image = cv_bridge.imgmsg_to_cv2(image, "rgb8")
	image = PIL.Image.fromarray(cv_image)

	# Find the bounds of the signal
	xmin, xmax, ymin, ymax = calculate_bounds(bounds)

	# Crop the image for the ROI
	cropped_roi = crop_image(image, xmin, xmax, ymin, ymax)

	roi_image.publish(cv_bridge.cv2_to_imgmsg(numpy.array(cropped_roi), "rgb8"))
	# Run the cropped image through the NN
	prediction = predict_light(cropped_roi)

	# Publish the prediction
	light_detected_pub.publish(traffic_light(recognition_result=prediction))
	print(rospy.Time.now(), detect_cnt, prediction)
	detect_cnt += 1


rospy.init_node('traffic_light_classifier', anonymous=True)

detect_cnt = 0
bounds = BBox(x=10, y=10, width=600, height=400)
predict_sub = rospy.Subscriber('/usb_cam/image_raw', Image, detect_signal, queue_size=1, buff_size=52428800)
light_detected_pub = rospy.Publisher(
	'light_color', traffic_light, queue_size=1)
roi_image = rospy.Publisher('roi_image', Image, queue_size=1)

rospy.spin()
