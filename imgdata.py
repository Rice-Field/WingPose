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

imgarray = np.zeros((20, 256, 256, 3), dtype=np.float32)


i = 0
os.chdir("./pics3")
for file in glob.glob("*.jpg"):
# for i in range(len(filenames)):
	if i == 19:
		break
	# print(i)
	# path = os.path.join('./pics/', filenames[i][0])
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
	
np.save("imgdataC_256-3", imgarray)