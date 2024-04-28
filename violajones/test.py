import cv2

palm_model = cv2.CascadeClassifier('D:/PROJECTS/Sandboil_Detection_and_Prediction/violajones/cascade_15Stages.xml')
list_path = 'D:/PROJECTS/Sandboil_Detection_and_Prediction/violajones/test-run/test-data/test-list.txt'

with open(list_path) as mynewfile:
    contents = mynewfile.read().splitlines()
    for a in contents:
        image = cv2.imread('D:/PROJECTS/Sandboil_Detection_and_Prediction/violajones/test-run/test-data/' + a + '.jpg', 0)

        scale_factor = 1.1
        min_neighbors = 1
        min_size = (20, 20)

        palms = palm_model.detectMultiScale(image, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size)
        for (x, y, w, h) in palms:
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 1, 0), 2)
            cv2.imwrite(('D:/PROJECTS/Sandboil_Detection_and_Prediction/violajones/test-run/results-test/' + str(a) + ".jpg"), image)
