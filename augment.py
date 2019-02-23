import os
import glob
import numpy as np
import pandas as pd
import cv2 as cv
import imutils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageFile

# start with simple augmentations
# 90* rotations, flips, noise and smoothing

imgdata = np.load("imgdataC_256.npy")
# hm = np.load("heatmap8_s1_64.npy")
hm = np.load("heatmap8_s3_128.npy")

nimgdata = np.zeros((3088,256,256,3), dtype=np.float32)
nhm = np.zeros((3088,128,128,8), dtype=np.float32)

# are hm channels flipping correctly?
j = 0
for i in range(0,1544,8):
	# print(j)
	nimgdata[i] = imgdata[j]
	nimgdata[i+1] = np.fliplr(nimgdata[i])
	nhm[i] = hm[j]
	nhm[i+1] = np.fliplr(nhm[i])

	# huge mess for flip and orig rotations, fix
	
	# nimgdata[i+2] = np.rot90(nimgdata[i])
	# nimgdata[i+3] = np.rot90(nimgdata[i+1])
	# nimgdata[i+4] = np.rot90(nimgdata[i+2])
	# nimgdata[i+5] = np.rot90(nimgdata[i+3])
	# nimgdata[i+6] = np.rot90(nimgdata[i+4])
	# nimgdata[i+7] = np.rot90(nimgdata[i+5])

	nhm[i+2] = np.rot90(nhm[i])
	nhm[i+3] = np.rot90(nhm[i+1])
	nhm[i+4] = np.rot90(nhm[i+2])
	nhm[i+5] = np.rot90(nhm[i+3])
	nhm[i+6] = np.rot90(nhm[i+4])
	nhm[i+7] = np.rot90(nhm[i+5])
	j += 1

for i in range(1544,3088):
	# nimgdata[i] += nimgdata[i-1544] + np.random.normal(0,1,(256,256,3))
	nhm[i] += nhm[i-1544]

# for i in range(8):
# 	plt.imshow(nimgdata[i].astype(np.uint8))
# 	plt.show()

# for i in range(8):
# 	plt.imshow(nimgdata[i+1544].astype(np.uint8))
# 	plt.show()

# np.save("AugimgdataC_256", nimgdata)
np.save("Augheatmap8_s3_128", nhm)

# for i in range(len(hm),len(hm)*2):
# 	nimgdata[i] = np.fliplr(imgdata[i])

# for i in range(len(hm)*2,len(nhm)):
# 	nimgdata[i] = np.rot90(imgdata[i])

# img = np.rot90(imgdata[0])
# hm1 = np.rot90(hm[0,:,:,0])

# img = imgdata[0]
# img = np.fliplr(imgdata[0])
# hm1 = np.fliplr(hm[0,:,:,0])

# plt.imshow(img.astype(np.uint8))
# plt.show()
# plt.imshow(hm1)
# plt.show()




# Show flip and rotations
#########################
# fig = plt.figure(figsize=(8,8))
# columns = 4
# rows = 2
# for i in range(1, columns*rows+1):
# 	if i == 5:
# 		img = np.fliplr(img)
# 	fig.add_subplot(rows, columns, i)
# 	plt.imshow(img.astype(np.uint8))
# 	img = np.rot90(img)
# plt.show()