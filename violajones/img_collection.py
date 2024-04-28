import cv2
import numpy
import urllib2 	

image_url = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n03160309#'
image_collections = urllib2.urlopen(image_url).read()

image_id=1

for i in image_collections.split('\n'):
  try:
      print('image from '+str(i))
      img_data = urllib2.urlopen(str(i)).read()
      fout = open("negative/"+str(image_id)+'.png','wb')
      fout.write(img_data)
      fout.close()

      img=cv2.imread("negative/"+str(image_id)+'.png', cv2.IMREAD_GRAYSCALE)
      resize_img = cv2.resize(img, (140,140))
      cv2.imwrite("negative/"+str(image_id)+'.png',resize_img)
      image_id = image_id + 1
      
  except Exception as Err:
  	print(str(Err))
