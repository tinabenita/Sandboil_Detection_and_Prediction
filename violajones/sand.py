import cv2
#import numpy 


#palm_model = cv2.CascadeClassifier('/home/avdesh/AditiEdge/data/cascade.xml')
palm_model = cv2.CascadeClassifier('D:/PROJECTS/Sandboil_Detection_and_Prediction/violajones/cascade_15Stages.xml')

list_path='D:/PROJECTS/Sandboil_Detection_and_Prediction/violajones/test-run/test-data/test-list.txt'
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
			gray = cv2.rectangle(image,(x,y),(x+w,y+h),(50,50,200),2)
			f=open('D:/PROJECTS/Sandboil_Detection_and_Prediction/violajones/test-run/predicted-test.txt','a')
			f.write('%i %i %i %i\n'%(x,y,w,h))
f.close()

#cam = cv2.VideoCapture(0)
