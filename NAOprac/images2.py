# -*- encoding: UTF-8 -*-
# Get some images from the NAO. Display or save them to disk.

import sys
import time
import numpy as np
import cv2
from naoqi import ALProxy


IP = "172.22.0.3"
PORT = 9559


# convert the image to a format OpenCV can use
def toCVImg(naoImage):

    #newImages = []

    #for naoImage in images:
	
	# Get the image size and pixel array.
	imageWidth = naoImage[0]
	imageHeight = naoImage[1]
	channels = naoImage[2]
	array = naoImage[6]
	camera = naoImage[7]
	#print(imageHeight, imageWidth, channels, camera)

	# Create a PIL Image from our pixel array.
	img = (np.reshape(
			np.frombuffer(
				array,
				dtype='%iuint8' % channels
			),
			(imageHeight, imageWidth, channels))
        )

	#newImages.append(img)
	return img

def toCVImg2(naoimg):
	width = 320
	height = 240
	image = np.zeros((height, width, 3), np.uint8)
	values = map(ord, list(naoimg[6]))
	i = 0
	for y in range(0, height):
		for x in range(0, width):
			image.itemset((y, x, 0), values[i + 0])
			image.itemset((y, x, 1), values[i + 1])
			image.itemset((y, x, 2), values[i + 2])
			i += 3
	return image

def showNaoImage():
	"""
	First get an image from Nao, then show it on the screen.
	"""
 
	client = "python_GVM"
	tracker = ALProxy("ALTracker", IP, PORT)
	camProxy = ALProxy("ALVideoDevice", IP, PORT)
	resolution = 0    # 0: 160x120, 1: 320x240, 2: 640x480, 3: 1280x960
	colorSpace = 13   # BGR (because OpenCV)
	fps = 30
	# if you only want one camera
	videoClient = camProxy.subscribeCamera(client, 0, resolution, colorSpace, fps)
	# if you want both cameras
	#videoClient = camProxy.subscribeCameras(client, [0, 1], [resolution, resolution], [colorSpace, colorSpace], fps)
	print videoClient
	print 'getting images in remote'
	num_images = 1000
	targetName = "RedBall"
	diameterOfBall = 0.04
	tracker.registerTarget(targetName, diameterOfBall)
	tracker.track(targetName)
	for i in range(0, num_images):
		print "getting image " + str(i)
		images = camProxy.getImageRemote(videoClient)
		#print images
		if images == None:
			print images
		imgs = toCVImg(images)

		
		p = tracker.getTargetPosition(0)
		print p
		# only relevant when using both cameras
		"""for camNum, img in enumerate(imgs):
			# Save the image.
			if camNum == 0:
				camName = "top"
				print("From top camera")
			if camNum == 1:
				camName = "bottom"
				print("From bottom camera")
			cv2.imshow("image", img)"""
			#cv2.imwrite("nao_recordings/image_%i_%s.png" % (i, camNum), img)

		# for single camera
		cv2.imshow("image", imgs)
		#cv2.imwrite("nao_recordings/image_%i.png" % i, imgs)
		cv2.waitKey(1)
		

	camProxy.unsubscribe(videoClient)


if __name__ == '__main__':
    naoImage = showNaoImage()