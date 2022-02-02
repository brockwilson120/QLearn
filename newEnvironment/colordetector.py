import cv2
from PIL import Image
import numpy as np

numpyColors = [
([0, 0, 200], [0, 0, 255]),
([200, 0, 0], [255, 0, 0]),
([200, 200, 200], [255, 255, 255]),
([0, 0, 0], [0, 0, 0])
]

class colorDetect:

	def __init__(self):
		self.detect()

	def detect(self):
		image = cv2.imread('genericName.png')
		neuroInput = []
		for lower, upper in numpyColors:
			upper = np.array(upper, dtype='uint8')
			lower = np.array(lower, dtype='uint8')
			mask = cv2.inRange(image, lower, upper)
			#output = cv2.bitwise_and(image, image, mask = mask)

			neuroInput.append(sum(sum(mask)) != 13575)
		return neuroInput

