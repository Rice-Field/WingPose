import os
import glob
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from shutil import copyfile

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

for i in range(len(filenames)):
	# if i == 19:
	# 	break
	if filenames[i][0][0] == '3':
		continue
	print(i)
	path1 = os.path.join('./pics/', filenames[i][0])
	path2 = os.path.join('./mywork/', filenames[i][0])
	copyfile(path1, path2)