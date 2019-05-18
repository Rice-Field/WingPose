import os
import glob
import numpy as np
import pandas as pd
import cv2 as cv
import imutils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# save filepath to variable for easier access
path = 'truewing.csv'
# read the data and store data in DataFrame titled melbourne_data
data = pd.read_csv(path)
points = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5',
				 'x6', 'y6', 'x7', 'y7', 'x8', 'y8']
name = ['filename']
X = data[points]
files = data[name]
numdata = X.values.tolist()
filenames = files.values.tolist()

# save filepath to variable for easier access
path = './contours/polygon.csv'
# read the data and store data in DataFrame titled melbourne_data
data = pd.read_csv(path)
name = ['filename']
X = data.loc[:, data.columns != 'filename']
files = data[name]
polygons = X.values.tolist()
filenames2 = files.values.tolist()

imgarray = np.zeros((142, 256, 256, 3), dtype=np.float32)

picNames = []

i = 0
os.chdir("./pics4")
for file in glob.glob("*.tif"):
# for i in range(len(filenames2)):
	print(i)

	picNames.append(file)
	# path = os.path.join('./pics4/', filenames2[i][0])
	path = file
	img = Image.open(path)
	# img = mpimg.imread(path)
	# img = img.convert("L")
	img = img.resize((256,256), Image.ANTIALIAS)
	img = np.asarray(img)
	imgarray[i] = img

	# print(imgarray[i].astype(uint8))
	# plt.imshow(imgarray[i].astype(np.uint8))
	# plt.show()
	# break
	i += 1

for file in glob.glob("*.jpg"):
	print(i)

	picNames.append(file)
	# path = os.path.join('./pics4/', filenames2[i][0])
	path = file
	img = Image.open(path)
	# img = mpimg.imread(path)
	# img = img.convert("L")
	img = img.resize((256,256), Image.ANTIALIAS)
	img = np.asarray(img)
	imgarray[i] = img

	# print(imgarray[i].astype(uint8))
	# plt.imshow(imgarray[i].astype(np.uint8))
	# plt.show()
	# break
	i += 1
	
np.save("pics4", imgarray)

pd_picNames = pd.DataFrame(picNames)
pd_picNames.to_csv('pics4.csv', index=False, header=False)