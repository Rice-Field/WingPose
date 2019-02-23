import os
import glob
import math
import numpy as np
import pandas as pd
import cv2 as cv
import imutils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageFile

# Convert heatmap into x,y coordinates for keypoint measurements
# simple methode: grab x,y of max in each heatmap
# better?: find center of each density

# Load our data
path = 'truewing.csv'
data = pd.read_csv(path)
points = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5',
                 'x6', 'y6', 'x7', 'y7', 'x8', 'y8']
name = ['filename']
X = data[points]
files = data[name]
numdata = X.values.tolist()
filenames = files.values.tolist()

# imgdata = np.load("AugimgdataC_256.npy")
imgdata = np.load("imgdataC_256-3.npy")
# hm = np.load("heatmap8_s1_64.npy")
# hm = np.load("trainhm.npy")
hm = np.load("./output/testhm128-2.npy")

def distance(pt1, pt2):
	return math.sqrt( (pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2 )

def angle(pt1,pt2):
    m1 = (pt1[1] - pt1[1])/1
    m2 = (pt2[1] - pt1[1])/(pt2[0]-pt1[0])

    tnAngle = (m1-m2)/(1+(m1*m2))
    tnAngle = math.atan(tnAngle)
    ang = tnAngle*180/math.pi
    ang *= (-1)
    return ang

def findangle(pt1,pt2):
    deltaX = pt2[0] - pt1[0]
    deltaY = pt2[1] - pt1[1]

    return math.atan2(deltaY,deltaX)*180/math.pi

# p1 is center
def anglebetween(p0, p1, p2):
	a = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
	b = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
	c = (p2[0]-p0[0])**2 + (p2[1]-p0[1])**2
	return math.acos( (a+b-c) / math.sqrt(4*a*b) ) * 180/math.pi

# find max points and store in a array
def maxpoint(hm1):
	return np.unravel_index(hm1.argmax(), hm1.shape)

def findJoints(hm8):
	joints = np.zeros((8,2), dtype=int)

	for i in range(8):
		index = maxpoint(hm8[:,:,i])

		# joints[i,0] = int(index[1] / 64 * 256) # x
		# joints[i,1] = int(index[0] / 64 * 256) # y
		# joints[i,0] = int(index[1] / 64 * 1600) # x
		# joints[i,1] = int(index[0] / 64 * 1200) # y

		joints[i,0] = int(index[1] / 128 * 256) # x
		joints[i,1] = int(index[0] / 128 * 256) # y
		# joints[i,0] = int(index[1] / 128 * 1600) # x
		# joints[i,1] = int(index[0] / 128 * 1200) # y

	return joints

# display joints
def drawjoints(img, joints):

	cv.circle(img, (joints[0][0],joints[0][1]), 3, (255,0,0), 1)
	cv.circle(img, (joints[1][0],joints[1][1]), 3, (0,255,0), 1)
	cv.circle(img, (joints[2][0],joints[2][1]), 3, (0,255,0), 1)
	cv.circle(img, (joints[3][0],joints[3][1]), 3, (0,255,0), 1)
	cv.circle(img, (joints[4][0],joints[4][1]), 3, (0,0,255), 1)
	cv.circle(img, (joints[5][0],joints[5][1]), 3, (0,0,255), 1)
	cv.circle(img, (joints[6][0],joints[6][1]), 3, (0,0,255), 1)
	cv.circle(img, (joints[7][0],joints[7][1]), 3, (255,0,0), 1)

	cv.line(img, (joints[0][0],joints[0][1]), (joints[7][0],joints[7][1]), (255,0,0), 1, cv.LINE_AA)
	cv.line(img, (joints[1][0],joints[1][1]), (joints[2][0],joints[2][1]), (0,255,0), 1, cv.LINE_AA)
	cv.line(img, (joints[1][0],joints[1][1]), (joints[3][0],joints[3][1]), (0,255,0), 1, cv.LINE_AA)
	cv.line(img, (joints[4][0],joints[4][1]), (joints[5][0],joints[5][1]), (0,0,255), 1, cv.LINE_AA)
	cv.line(img, (joints[4][0],joints[4][1]), (joints[6][0],joints[6][1]), (0,0,255), 1, cv.LINE_AA)

	return img


allmeasures = []
for i in range(20):
	measure = []
	# img = cv.imread(('./pics/' + filenames[i+181][0]),cv.IMREAD_COLOR)
	# img = cv.imread(('./pics/' + filenames[i][0]),cv.IMREAD_COLOR)
	# img = np.copy(imgdata[i+180])
	img = np.copy(imgdata[i])
	measure.append(filenames[i][0])

	joints = findJoints(np.copy(hm[i]))
	# print(joints)
	measure.append(distance(joints[0],joints[7]))
	measure.append(0)
	measure.append(distance(joints[1],joints[2]))
	measure.append(0)
	# measure.append(anglebetween(joints[2],joints[1],joints[3]))
	measure.append(distance(joints[4],joints[5]))
	measure.append(0)
	# measure.append(anglebetween(joints[2],joints[1],joints[3]))

	img = drawjoints(img, joints)

	plt.imshow(img.astype(np.uint8))
	plt.show()

	allmeasures.append(measure)

column_name = ['filename', 'Major', 'Minor', 'Length 1', 'Angle 1', 'Length 2', 'Angle 2']
data = pd.DataFrame(allmeasures, columns=column_name)
data.to_csv('measurements.csv', index=None)
print('Successfully created csv.')