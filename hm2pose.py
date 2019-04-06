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
path = 'truewing2.csv'
data = pd.read_csv(path)
points = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5',
                 'x6', 'y6', 'x7', 'y7', 'x8', 'y8']
name = ['filename']
X = data[points]
files = data[name]
numdata = X.values.tolist()
filenames = files.values.tolist()

imgdata = np.load("imgdataPoly_256-3.npy")
# imgdata = np.load("contourImg.npy")
# hm = np.load("heatmap8_s1_64.npy")
hm = np.load("./output/testhmC2.npy")
# hm = np.load("./output/testhmR2.npy")
# hm = np.load("contour_s3_128.npy")
# hm = np.load("./output/testhmRes4.npy")

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

def avgdensity(hm):
	n_points = 4
	avgx = 0
	avgy = 0
	for i in range(n_points):
		# print(np.max(hm))
		index = maxpoint(hm)
		avgx += index[0]
		avgy += index[1]
		hm[index[0]][index[1]] = 0

	avgx /= n_points
	avgy /= n_points

	return [avgx, avgy]

def findDensityJoints(hm8):
	joints = np.zeros((9,2), dtype=int)

	for i in range(8):
		index = avgdensity(hm8[:,:,i])

		# joints[i,0] = round(index[1] / 64 * 256) # x
		# joints[i,1] = round(index[0] / 64 * 256) # y
		# joints[i,0] = round(index[1] / 64 * 1600) # x
		# joints[i,1] = round(index[0] / 64 * 1200) # y

		joints[i,0] = round(index[1] / 128 * 256) # x
		joints[i,1] = round(index[0] / 128 * 256) # y
		# joints[i,0] = round(index[1] / 128 * 1600) # x
		# joints[i,1] = round(index[0] / 128 * 1200) # y

	return joints

# find max points and store in a array
def maxpoint(hm1):
	return np.unravel_index(hm1.argmax(), hm1.shape)

def findJoints(hm8):
	joints = np.zeros((9,2), dtype=int)

	for i in range(8):
		index = maxpoint(hm8[:,:,i])

		# joints[i,0] = round(index[1] / 64 * 256) # x
		# joints[i,1] = round(index[0] / 64 * 256) # y
		# joints[i,0] = round(index[1] / 64 * 1600) # x
		# joints[i,1] = round(index[0] / 64 * 1200) # y

		joints[i,0] = round(index[1] / 128 * 256) # x
		joints[i,1] = round(index[0] / 128 * 256) # y
		# joints[i,0] = round(index[1] / 128 * 1600) # x
		# joints[i,1] = round(index[0] / 128 * 1200) # y

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
	cv.circle(img, (joints[8][0],joints[8][1]), 3, (255,165,0), 1)

	cv.line(img, (joints[0][0],joints[0][1]), (joints[7][0],joints[7][1]), (255,0,0), 1, cv.LINE_AA)
	cv.line(img, (joints[1][0],joints[1][1]), (joints[2][0],joints[2][1]), (0,255,0), 1, cv.LINE_AA)
	cv.line(img, (joints[1][0],joints[1][1]), (joints[3][0],joints[3][1]), (0,255,0), 1, cv.LINE_AA)
	cv.line(img, (joints[4][0],joints[4][1]), (joints[5][0],joints[5][1]), (0,0,255), 1, cv.LINE_AA)
	cv.line(img, (joints[4][0],joints[4][1]), (joints[6][0],joints[6][1]), (0,0,255), 1, cv.LINE_AA)
	cv.line(img, (joints[2][0],joints[2][1]), (joints[8][0],joints[8][1]), (255,165,0), 1, cv.LINE_AA)

	return img

def vectcontour(hm):

	hm /= np.max(hm)

	w,h = (hm.shape)
	x, y = np.mgrid[0:128, 0:128]

	dy, dx = np.gradient(hm)

	mag = (dy**2 + dx**2)**(0.5)
	mag /= np.max(mag)

	invmag = (mag - 1) * (-1)

	peri = np.where(invmag + hm > 1.4, invmag, 0)
	peri /= np.max(peri)

	return peri

def findperp(joints, peri):
	# find equation of line for major segment
	denom = (joints[7][1] - joints[0][1])
	if denom == 0:
		denom = 0.000000000001
	m1 = (joints[7][0] - joints[0][0]) / denom
	b1 = -m1 * joints[0][1] + joints[0][0]

	# find line perpendicular to major segment
	m2 = -1/m1
	b2 = -m2 * joints[2][1] + joints[2][0]

	# find intersection of both
	n = b2 - b1
	z = m1 - m2
	if z == 0:
		z = 0.000000000001
	x = n/z
	y = m2 * x + b2

	x = round(x).astype(int)
	y = round(y).astype(int)

	# move away from source
	dx = joints[2][1] - x
	if dx < 0:
		x += 1
		dx = 1
	else:
		x += -1
		dx = -1

	dy = joints[2][0] - y
	if dy < 0:
		y += 1
		dy = 1
	else:
		y += -1
		dy = -1
	# search till contact with perimeter to for minor segment
	search = True
	# print(np.max(peri))
	# print([y,x])
	while(search):
		y2 = round(m2 * x + b2).astype(int)
		x2 = round((y - b2)/ m2).astype(int)

		if y == y2:
			x += dx

		elif x == x2:
			y += dy

		else:
			x += dx
			y += dy

		if peri[x,y] > 0.5:
			search = False
			break
		# print(peri[y,x])
			
	joints[8,0] = y
	joints[8,1] = x

	print(joints[8])
	return joints


images = np.zeros((10,256,256,3))
allmeasures = []
for i in range(10):
	measure = []
	# img = cv.imread(('./pics/' + filenames[i+288][0]),cv.IMREAD_COLOR)
	# img = cv.imread(('./pics/' + filenames[i][0]),cv.IMREAD_COLOR)
	img = np.copy(imgdata[i+288])
	# img = np.copy(imgdata[i])
	# measure.append(filenames[i][0])

	joints = findDensityJoints(np.copy(hm[i,:,:,0:8]))
	# print(joints)
	# measure.append(distance(joints[0],joints[7]))
	# measure.append(0)
	# measure.append(distance(joints[1],joints[2]))
	# measure.append(0)
	# # measure.append(anglebetween(joints[2],joints[1],joints[3]))
	# measure.append(distance(joints[4],joints[5]))
	# measure.append(0)
	# measure.append(anglebetween(joints[2],joints[1],joints[3]))


	cont = np.copy(hm[i,:,:,8])
	peri = vectcontour(cont)
	peri = Image.fromarray(peri)
	peri = peri.resize((256,256), Image.ANTIALIAS)
	peri = np.asarray(peri)

	# plt.imshow(peri)
	# plt.show()

	img[:,:,1] = np.where(peri > .3, 255, img[:,:,1])
	img[:,:,0] = np.where(peri > .3, 255, img[:,:,0])
	img[:,:,2] = np.where(peri > .3, 0, img[:,:,2])

	joints = findperp(joints, peri)

	img = drawjoints(img, joints)

	images[i] = np.copy(img)
	# break

	# plt.imshow(hm[i,:,:,8])
	# plt.show()
	# break

	# for j in range(8):
	# 	plt.imshow(hm[0,:,:,j])
	# 	plt.show()


	# allmeasures.append(measure)

# column_name = ['filename', 'Major', 'Minor', 'Length 1', 'Angle 1', 'Length 2', 'Angle 2']
# data = pd.DataFrame(allmeasures, columns=column_name)
# data.to_csv('measurements.csv', index=None)
# print('Successfully created csv.')

fig = plt.figure(figsize=(256,256))
columns = 5
rows = 2
for j in range(1, 11):
    fig.add_subplot(rows, columns, j)
    plt.axis('off')
    plt.imshow(images[j-1].astype(np.uint8))
fig.tight_layout()
plt.show()
