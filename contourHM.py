import json
import os
import glob
import imutils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
import sys
import scipy.ndimage as ndimage
from sklearn.preprocessing import normalize
from pylab import *

ImageFile.LOAD_TRUNCATED_IMAGES = True

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

json_file = open("wing1.json").read()

parsed_json = json.loads(json_file)
list = parsed_json["annotations"][1]["segmentation"][0]

polygon = []
for i in range(0,len(list), 2):
    point = [list[i], list[i+1]]
    polygon.append(point)


def toSmall(polygon):
    for k in range(len(polygon)):
        polygon[k][0] = int(polygon[k][0]/oimg_h * nimg_h)
        polygon[k][1] = int(polygon[k][1]/oimg_w * nimg_w)

    return polygon

def normalize(array):
    maxim = np.max(array)
    array /= maxim
    return array

polygon = toSmall(polygon)

# print(polygon)

contourList = []
for i in range(0, len(polygon)):
    j = i + 1

    if (i == len(polygon) - 1):
        j = 0

    denom = (polygon[j][0] - polygon[i][0])
    if denom == 0:
        denom = 0.000000000001

    slope = (polygon[j][1] - polygon[i][1]) / denom
    b = - slope * polygon[i][0] + polygon[i][1]

    contourList.append(polygon[i])

    for k in range(min(polygon[i][0], polygon[j][0]), max(polygon[i][0], polygon[j][0])):
        end = int(slope * k + b)
        point = [k, end]
        contourList.append(point)

    for k in range(min(polygon[i][1], polygon[j][1]), max(polygon[i][1], polygon[j][1])):
        end = int((k - b)/ slope)
        point = [end, k]
        contourList.append(point)

print(contourList)

hm = np.zeros((nimg_h, nimg_w), dtype=np.float32)

# for i in range(0, len(contourList)):
#     y = contourList[i][1]
#     x = contourList[i][0]
#     hm[y][x] = 255


# plt.show()

wingImage = Image.open("2-D-F4-L4.jpg")
wingImage = wingImage.convert("L")
wingImage = wingImage.resize((128, 128), Image.ANTIALIAS)
wingImage = np.asarray(wingImage)

# plt.imshow(wingImage)
#
# finished = wingImage + hm
# plt.imshow(finished)
# plt.show()

sigma = 3



for i in range(len(contourList)):
    y0 = contourList[i][1]
    x0 = contourList[i][0]

    var = pow(np.e, (-((xmap-x0)**2 + (ymap-y0)**2)/(2*sigma*sigma))) / (math.sqrt(2*np.pi)*sigma)
    hm[:][hm < var] = var[hm < var]

hm = normalize(hm)
# visualize in image
hm *= 255


plt.imshow(hm)
plt.show()
