import os
import glob
import numpy as np
import pandas as pd
import cv2 as cv
import imutils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageFile


os.chdir("./lastbatch")
for file in glob.glob("*.tif"):