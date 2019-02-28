# WingPose
Pose estimation of mosquito wings with deep learning. Using an autoencoder architecture, heatmaps can be generated from the image to give the location of keypoints used in the measurement wings.

### Prototype

![Prototype](images/5prototype_img1.png)

### Current examples

![Examples](images/pose_set2.png)

### Final Goal

![Goal](images/wingpose.png)

## Dataset
The dataset was created from a lab interested in the development of adult mosquitos. Currently 8 x,y coordinate values are used for the dataset. In the future the perimeter or area of the wing will be included as a contour or object pixel classification. A link to the set of images will be included shortly.

### Order of keypoints

Points</br>
1: </br>
2: </br>
3: </br>
4: </br>
5: </br>
6: </br>
7: </br>
8: </br>

## Things To Do
- Finish keypoint labels
- Robust image augmentations for the dataset
  * Decide between offline and online augments
- Labeling for wing contour or segmentation
- Improve model architecture
  * PoseNet, ResNet, DenseNet
  * One network for keypoints and contour detection?

## Steps
1. Label with [imglab.ml](imglab.ml)
2. Convert xml file to csv with xml2csv.py</br>
  &nbsp;&nbsp;- json files will be used for contour polygons</br>
3. Use heatmap.py to generate GT labels for keypoints
4. imgdata.py to store all images as a numpy array
5. Train with autoencoder.py and save weights
6. Run hm2pose.py to interpret heatmaps to keypoints and display poses
