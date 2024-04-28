import cv2
#import numpy 


palm_model = cv2.CascadeClassifier('D:/PROJECTS/Sandboil_Detection_and_Prediction/violajones/cascade_15Stages.xml')

list_path='D:/PROJECTS/Sandboil_Detection_and_Prediction/violajones/test-run/test-data/test-list.txt'
#output_path='D:/PROJECTS/Sand_Boil_Detection/code_data/code_data/ViolaJones/pos/result'
with open(list_path) as mynewfile:
	contents=mynewfile.read().splitlines()
	for a in contents:
		image = cv2.imread('D:/PROJECTS/Sandboil_Detection_and_Prediction/violajones/test-run/test-data/'+a+'.jpg',0)
		#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		scale_factor = 1.1
		min_neighbors = 1
		min_size = (20, 20)


		palms = palm_model.detectMultiScale(image, scaleFactor = scale_factor, minNeighbors = min_neighbors, minSize = min_size)
		for (x,y,w,h) in palms:
			#gray = cv2.rectangle(image,(x,y),(x+w,y+h),(50,50,200),2)
			gray = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 2)
			cv2.imwrite(('D:/PROJECTS/Sandboil_Detection_and_Prediction/violajones/test-run/results-test/' + str(a) + ".jpg"),image)