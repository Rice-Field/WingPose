# WingPose
Pose estimation of mosquito wings with deep learning. Using an autoencoder architecture, heatmaps can be generated from the image to give the location of keypoints used in the measurement of wings.

### Keypoint Detection

![Keypoint](images/5prototype_img1.png)

### Contour Detection
We calculate the gradient vector field of the output heatmap to retrieve a sharp contour.
<img src="images/gradient_field.png" alt="drawing" width="800"/>

### Model Output

<img src="images/output2.png" alt="drawing" width="500"/>

## Dataset
The dataset was created from a lab interested in the development of adult mosquitos. Currently 8 x,y coordinate values are used for the dataset. In the future the perimeter or area of the wing will be included as a contour or object pixel classification. A link to the set of images will be included shortly.

### Order of keypoints

<img src="images/newpoints.PNG" alt="drawing" width="700"/>

## Steps
1. Label with [imglab.ml](imglab.ml)
2. Convert xml file to csv with xml2csv.py</br>
  &nbsp;&nbsp;- json files will be used for contour polygons</br>
3. Use heatmap.py to generate GT labels for keypoints
4. imgdata.py to store all images as a numpy array
5. Train with autoencoder.py and save weights
6. Run hm2pose.py to interpret heatmaps to keypoints and display poses
