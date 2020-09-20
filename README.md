# WingPose
Pose estimation of mosquito wings with deep learning. Using an autoencoder architecture, heatmaps can be generated from the image to give the location of keypoints and contour used in the measurement of wings.

### Desired measurements

<img src="images/measure.png" alt="drawing" width="500"/>

### Keypoint Detection
We apply gaussian smoothing to both keypoints and the contour for heatmap training.
![Keypoint](images/5prototype_img1.png)

### Contour Detection
To retrieve a sharp contour we calculate the gradient vector field of the output heatmap.
<img src="images/gradient_field.png" alt="drawing" width="700"/>

### Model Output
<img src="images/output3.png" alt="drawing" width="500"/>

## Dataset
The dataset was created from a lab interested in the development of adult mosquitos. Currently 8 x,y coordinate values are used for the dataset. In the future the perimeter or area of the wing will be included as a contour or object pixel classification. A link to the set of images will be included shortly.

#### Order of keypoints
<img src="images/newpoints.PNG" alt="drawing" width="700"/>

#### Contour annotation
<img src="images/Figure_2-2.png" alt="drawing" width="300"/>

## Steps
1. Label with [imglab.ml](imglab.ml)
2. Convert xml file to csv with xml2csv.py</br>
  &nbsp;&nbsp;- json files will be used for contour polygons</br>
3. Use heatmap.py to generate GT labels for keypoints
4. imgdata.py to store all images as a numpy array
5. Train with autoencoder.py and save weights
6. Run hm2pose.py to interpret heatmaps to keypoints and display poses
