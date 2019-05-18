import os
import glob
import imutils
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from PIL import Image
import sys
import math
import scipy.ndimage as ndimage
from sklearn.preprocessing import normalize
# from pylab import *

oimg_h = 1600
oimg_w = 1200
nimg_h = 128
nimg_w = 128

xmap = np.zeros((nimg_h,nimg_w))
for i in range(nimg_h):
    for j in range(nimg_w):
        xmap[i][j] = j

ymap = np.zeros((nimg_h,nimg_w))
for i in range(nimg_h):
    for j in range(nimg_w):
        ymap[i][j] = i

# plt.imshow(xmap)
# plt.show()
# plt.imshow(ymap)
# plt.show()

def pointCoord(numdata):
    points = []
    for k in range(len(numdata)):
        points.append([])
        for i in range(0,16,2):
            points[-1].append([numdata[k][i], numdata[k][i+1]])

    return points

def toSmall(points):

    for k in range(len(points)):
        for i in range(0,16,2):
            points[k][i] = round(points[k][i]/oimg_h * nimg_h)
            points[k][i+1] = round(points[k][i+1]/oimg_w * nimg_w)

    return points

# Calculates gaussian values
def gaussian(sigma, x0, y0, x, y):
    n = pow(np.e, (-((x-x0)**2 + (y-y0)**2)/(2*sigma*sigma)))
    return (n/(math.sqrt(2*np.pi)*sigma))

# normalize kernel to total to 1
def normalize(array):
    maxim = np.max(array)
    array /= maxim
    return array

def genheatmap(sigma, point, width, height):
    hm = np.zeros((height,width))
    x0 = point[0]
    y0 = point[1]

    hm += pow(np.e, (-((xmap-x0)**2 + (ymap-y0)**2)/(2*sigma*sigma))) / (math.sqrt(2*np.pi)*sigma)

    # for i in range(width):
    #     for j in range(height):
    #         hm[j,i] = gaussian(sigma, point[0], point[1], i, j)

    hm = normalize(hm)
    # visualize in image
    # hm *= 255

    return hm

# save filepath to variable for easier access
path = 'truewing2.csv'
# read the data and store data in DataFrame titled data
data = pd.read_csv(path)
points = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5',
                 'x6', 'y6', 'x7', 'y7', 'x8', 'y8']
name = ['filename']
X = data[points]
files = data[name]
numdata = X.values.tolist()
filenames = files.values.tolist()
# print(filenames[0])

numdata = np.asarray(numdata)
numdata = toSmall(numdata)
points = pointCoord(numdata)
# print(points[0][0])

hm = np.zeros((298, nimg_h, nimg_w, 8), dtype=np.float32)

# hm[0][numdata[0][1]][numdata[0][0]] = 255
# print([numdata[0][0], numdata[0][1]])

# Determine standard deviation for gaussian filtering
sigma = 3

for k in range(298):
    for i in range(8):
        hm[k,:,:,i] = genheatmap(sigma, points[k][i], nimg_w, nimg_h)
    print(k)

np.save("heatmap8_s3_128-R", hm)

############################
# Overlay heatmap onto image
############################
# for i in range(7):
#     hm[0,:,:,0] += genheatmap(sigma, points[0][i], nimg_w, nimg_h)
# path = os.path.join('./pics/', filenames[0][0])
# img = Image.open(path)
# img = img.convert("L")
# img = img.resize((nimg_w,nimg_h), Image.ANTIALIAS)
# img = np.asarray(img)

# imgarr = np.zeros((nimg_h,nimg_w))
# imgarr += img

# imgarr += hm[0,:,:,0]*255

# # plt.imshow(imgarr.astype(np.uint8))
# plt.imshow(hm[0,:,:,0])
# plt.show()
############################

