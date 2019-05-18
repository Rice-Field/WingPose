import os
import sys
import glob
import math
import numpy as np
import pandas as pd
import cv2 as cv
import imutils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageFile, ImageDraw, PILLOW_VERSION
import imgaug as ia
from imgaug import augmenters as iaa

# Convert heatmap into x,y coordinates for keypoint measurements
# simple methode: grab x,y of max in each heatmap
# better?: find center of each density

oimg_h = 1600
oimg_w = 1200

kimg_h = 256
kimg_w = 256

nimg_h = 128
nimg_w = 128


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
# hm = np.load("./output/testhmR2.npy")
# hm = np.load("contour_s3_128.npy")
# hm = np.load("./output/testhmRes4.npy")

# hm = np.load("./output/testhmT9.npy")
hm = np.load("./output/bigtesthm2.npy")

# hm3 = np.load("heatmap8_s3_128-R.npy")
# hm2 = np.load("contourHM_128_s2.npy")

# hm = np.zeros((298, 128, 128, 9), dtype = np.float32)
# hm[:,:,:, 0:8] += hm3
# hm[:,:,:, 8] += hm2

# pic4Names = pd.Series.from_csv('./pics4/pics4.csv')
pic4Names = pd.read_csv('./pics4/pics4.csv', header=None)
pic4Names = pic4Names[0].tolist()
# print(pic4Names)

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

		joints[i,0] = round(index[1] / 128 * oimg_h) # x
		joints[i,1] = round(index[0] / 128 * oimg_w) # y

	return joints

# find max points and store in a array
def maxpoint(hm1):
	return np.unravel_index(hm1.argmax(), hm1.shape)

def findJoints(hm8):
	joints = np.zeros((9,2), dtype=int)

	for i in range(8):
		index = maxpoint(hm8[:,:,i])

		joints[i,0] = round(index[1] / 128 * oimg_h) # x
		joints[i,1] = round(index[0] / 128 * oimg_w) # y

	return joints

# display joints
def drawjoints(img, joints):
	radius = 3
	thick = 5

	cv.circle(img, (joints[0][0],joints[0][1]), thick, (255,0,0), radius)
	cv.circle(img, (joints[1][0],joints[1][1]), thick, (0,255,0), radius)
	cv.circle(img, (joints[2][0],joints[2][1]), thick, (0,255,0), radius)
	cv.circle(img, (joints[3][0],joints[3][1]), thick, (0,255,0), radius)
	cv.circle(img, (joints[4][0],joints[4][1]), thick, (0,0,255), radius)
	cv.circle(img, (joints[5][0],joints[5][1]), thick, (0,0,255), radius)
	cv.circle(img, (joints[6][0],joints[6][1]), thick, (0,0,255), radius)
	cv.circle(img, (joints[7][0],joints[7][1]), thick, (255,0,0), radius)
	cv.circle(img, (joints[8][0],joints[8][1]), thick, (255,165,0), radius)

	cv.line(img, (joints[0][0],joints[0][1]), (joints[7][0],joints[7][1]), (255,0,0), radius, cv.LINE_AA)
	cv.line(img, (joints[1][0],joints[1][1]), (joints[2][0],joints[2][1]), (0,255,0), radius, cv.LINE_AA)
	cv.line(img, (joints[1][0],joints[1][1]), (joints[3][0],joints[3][1]), (0,255,0), radius, cv.LINE_AA)
	cv.line(img, (joints[4][0],joints[4][1]), (joints[5][0],joints[5][1]), (0,0,255), radius, cv.LINE_AA)
	cv.line(img, (joints[4][0],joints[4][1]), (joints[6][0],joints[6][1]), (0,0,255), radius, cv.LINE_AA)
	cv.line(img, (joints[2][0],joints[2][1]), (joints[8][0],joints[8][1]), (255,165,0), radius, cv.LINE_AA)

	return img

def vectcontour(hm):

	hm /= np.max(hm)
	# plt.imshow(hm)
	# plt.show()

	w,h = (hm.shape)
	x, y = np.mgrid[0:128, 0:128]

	dy, dx = np.gradient(hm)

	mag = (dy**2 + dx**2)**(0.5)
	mag /= np.max(mag)

	invmag = (mag - 1) * (-1)

	cont = np.where((invmag + hm) > 1.4, invmag, 0)
	cont /= np.max(cont)
	# plt.imshow(cont)
	# plt.show()

	return cont

def findperp(joints, peri):
	# find equation of line for major segment
	inf = 0
	denom = (joints[7][1] - joints[0][1])
	if denom == 0:
		denom = 0.000000000001
	m1 = (joints[7][0] - joints[0][0]) / denom
	if m1 == 0:
		inf = 1
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

	found = 0
	xx = x
	yy = y
	# search till contact with perimeter to for minor segment
	while(True):

		# if slope is not infinite
		if inf == 0:
			# slope means y changes less than x
			# iterate over x
			if abs(m2) <= 1:
				y = round(m2 * x + b2).astype(int)

				if peri[x,y] > 0.5:
					# break
					xx = x
					yy = y
					found = 1
				
				if found == 1 and peri[x,y] == 0:
					break

				x += dx
			# x changes greater than y
			# iterate over y
			else: 	
				x = round((y - b2)/ m2).astype(int)
				if peri[x,y] > 0.5:
					# break
					xx = x
					yy = y
					found = 1
				
				if found == 1 and peri[x,y] == 0:
					break

				y += dy
		# x never changes, search up or down
		else:
			if peri[x,y] > 0.1:
				# break
				xx = x
				yy = y
				found = 1
				
			if found == 1 and peri[x,y] == 0:
				break

			y += dy
			
	# joints[8,0] = round((y + yy) / 2)
	# joints[8,1] = round((x + xx) / 2)
	joints[8,0] = yy
	joints[8,1] = xx

	return joints

def floodfill(matrix, x, y):

	#"hidden" stop clause - not reinvoking for "c" or "b", only for "a".
	if matrix[x,y] == 0:  
		matrix[x,y] = 1 
		#recursively invoke flood fill on all surrounding cells:
		if x > 0:
			matrix = floodfill(matrix, x-1, y)
		if x < 127:
			matrix = floodfill(matrix, x+1, y)
		if y > 0:
			matrix = floodfill(matrix, x, y-1)
		# if y < len(matrix[0]) - 1:
		if y < 127:
			matrix = floodfill(matrix, x, y+1)

	return matrix


seq = iaa.Sequential([
    iaa.EdgeDetect(alpha=(1.0, 1.0)),
    ])

seq_det = seq.to_deterministic()


if not os.path.exists('./pics4/measure/'):
    os.makedirs('./pics4/measure/')
images = np.zeros((10,oimg_w,oimg_h,3))
# img = np.copy(imgdata[288])
i = 0
allmeasures = []
os.chdir("./pics4")
for file in glob.glob("*.jpg"):
# for i in range(10):
	print(i)
	# if ( i < 13 ):
	# 	i += 1
	# 	continue

	measure = []
	print(file)
	# img = cv.imread('./pics/' + filenames[i+288][0]),cv.IMREAD_COLOR)
	# img = cv.imread('./pics/' + filenames[i][0]),cv.IMREAD_COLOR)
	img = cv.imread(pic4Names[i], cv.IMREAD_COLOR)
	oimg_w = img.shape[0]
	oimg_h = img.shape[1]
	# img = cv.imread(file)

	# img = np.copy(imgdata[i+288])
	# img = np.copy(imgdata[i])
	# measure.append(filenames[i][0])

	joints = findJoints(np.copy(hm[i,:,:,0:8]))

	cont = np.copy(hm[i,:,:,8])
	cont = vectcontour(cont)
	cont2 = Image.fromarray(cont)
	cont2 = cont2.resize((oimg_h,oimg_w), Image.ANTIALIAS)
	cont2 = np.asarray(cont2)
	cont2 = np.where(cont2 > .5, 1, 0)

	# if (file == "4-B-F4-R4.jpg"):
	# 	plt.imshow(cont)
	# 	plt.show()
		
	# print(img.shape)
	# print(cont.shape)
	# print(images[0].shape)
	img[:,:,1] = np.where(cont2 > .3, 255, img[:,:,1])
	img[:,:,0] = np.where(cont2 > .3, 255, img[:,:,0])
	img[:,:,2] = np.where(cont2 > .3, 0, img[:,:,2])

	joints = findperp(joints, cont2)

	img = drawjoints(img, joints)

	imager = Image.fromarray(img)
	pathy = file.split('.')
	imager.save('./measure/' + pathy[0] + '.png')
	# imager.show()

	# break
	# if i < 10:
	# 	images[i] = np.copy(img)
	i += 1
	
	# measure.append(distance(joints[0],joints[7]))
	# measure.append(distance(joints[2],joints[8]))
	# measure.append(distance(joints[1],joints[2]))
	# measure.append(anglebetween(joints[2],joints[1],joints[3]))
	# measure.append(distance(joints[4],joints[5]))
	# measure.append(anglebetween(joints[2],joints[1],joints[3]))

	# # Calc Perimeter in bogus way
	# conti = np.zeros((1,oimg_w,oimg_h))
	# conti[0] = cont2
	# edges = seq_det.augment_images(conti) 
	# # plt.imshow(edges[0])
	# # plt.show()
	# (values,counts) = np.unique(edges[0], return_counts=True)

	# plength = (oimg_h*oimg_w - np.max(counts))/4
	# # Perimeter
	# measure.append(plength)

	# # Calc area with flood fill on small img then expand to correct size and bin
	# x = round(joints[1][1]/oimg_w*nimg_w).astype(int)
	# y = round(joints[1][0]/oimg_h*nimg_h).astype(int)

	# cont = np.where(cont > .1, 1, 0)
	# area = floodfill(cont, x, y)
	# area = Image.fromarray(area.astype(np.uint8))
	# area = area.resize((oimg_h,oimg_w), Image.ANTIALIAS)
	# area = np.asarray(area)

	# (values2,counts2) = np.unique(area, return_counts=True)
	# # Area
	# measure.append(counts2[1])
	# # break
	# allmeasures.append(measure)

# os.chdir("..")
# column_name = ['filename', 'Major', 'Minor', 'Length 1', 'Angle 1', 'Length 2', 'Angle 2', 'Perimeter', 'Area']
# data = pd.DataFrame(allmeasures, columns=column_name)
# data.to_csv('./pics4/measure/measurements.csv', index=None)
# print('Successfully created csv.')

# fig = plt.figure(figsize=(256,256))
# columns = 5
# rows = 2
# for j in range(1, 11):
#     fig.add_subplot(rows, columns, j)
#     plt.axis('off')
#     plt.imshow(images[j-1].astype(np.uint8))
# fig.tight_layout()
# plt.show()
