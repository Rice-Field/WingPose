import imgaug as ia
from imgaug import augmenters as iaa
import random
import numpy as np
import pandas as pd
import time
ia.seed(int((time.time()*1000)%100000))

oimg_h = 1600
oimg_w = 1200
nimg_h = 256
nimg_w = 256

def toSmall(points):

    for k in range(len(points)):
        for i in range(0,16,2):
            points[k][i] = round(points[k][i]/oimg_h * nimg_h)
            points[k][i+1] = round(points[k][i+1]/oimg_w * nimg_w)

    return points

def pointCoord(numdata):
    points = []
    for k in range(len(numdata)):
        points.append([])
        for i in range(0,16,2):
            points[-1].append([numdata[k][i], numdata[k][i+1]])

    return points

def normalize(array):
    maxim = np.max(array)
    array /= maxim
    return array

# save filepath to variable for easier access
path = 'truewing.csv'
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

# images = np.random.randint(0, 100, (4, 128, 128, 3), dtype=np.uint8)
# images = np.random.uniform(0,1,[4, 128, 128, 3]).astype(np.float32)
images = np.load("imgdataC_256-3.npy")
images = images[0:4]/255

heatmaps = np.load("heatmap8_s3_128_R.npy")
heatmaps = heatmaps[0:4, :, :, 0:3]

# Generate random keypoints.
# The augmenters expect a list of imgaug.KeypointsOnImage.
i = 0
keypoints_on_images = []
for image in images:
    height, width = image.shape[0:2]
    keypoints = []
    for k in range(8):
        x = points[i][k][0]
        y = points[i][k][1]
        keypoints.append(ia.Keypoint(x=x, y=y))
    keypoints_on_images.append(ia.KeypointsOnImage(keypoints, shape=image.shape))
    i += 1

seq = iaa.Sequential([
    # iaa.GaussianBlur((0, 0.1)), 
    iaa.Crop(px=(0, 10)), # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.Affine(scale=(0.75, 1.25), translate_percent={"x": (-0.25, 0.25), "y": (-0.25, 0.25)}, rotate=(-30, 30)),
    # iaa.Pad(pad_mode=ia.ALL, pad_cval=(0, 255))
    ])

seq_det = seq.to_deterministic() # call this for each batch again, NOT only once at the start

# augment keypoints and images
images_aug = seq_det.augment_images(images) 
keypoints_aug = seq_det.augment_keypoints(keypoints_on_images)
heatmaps_aug = seq_det.augment_images(heatmaps) 

# Example code to show each image and print the new keypoints coordinates
for img_idx, (image_before, image_after, keypoints_before, keypoints_after, heatmap_before, heatmap_after) in enumerate(zip(images, images_aug, keypoints_on_images, keypoints_aug, heatmaps, heatmaps_aug)):
    image_before = keypoints_before.draw_on_image(image_before)
    image_after = keypoints_after.draw_on_image(image_after)
    ia.imshow(np.concatenate((image_before, image_after), axis=1)) # before and after
    # ia.imshow(np.concatenate((normalize(heatmap_before), normalize(heatmap_after)), axis=1))