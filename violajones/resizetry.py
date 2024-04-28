#interate over files in a directory and resize each into 140x140 size

import cv2
import numpy
import os

image_id=1

for filename in os.listdir(images):

	if filename.endswith(".png"):
	
		img=cv2.imread("tile"+str(image_id)+'.png', cv2.IMREAD_GRAYSCALE)
		resize_img = cv2.resize(img, (140,140))
		cv2.imwrite("tile"+str(image_id)+'.png',resize_img)
		image_id = image_id + 1
	except Exception as Err:
		print(str(Err))
