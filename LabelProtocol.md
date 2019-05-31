# Labeling Protocol
By Jonathon Rice

If you ever need guidance ask Natasha

## Website: https://imglab.ml
Any other platform is acceptable as long as the image (X,Y) coordinates are saved

- TIFF images do not work with this website

### Goal:
Decompose measurements into (X,Y) points of the image
- Allows for both machine learning and creating measurements in one

From: Old Measurements | To: A Contour and 9 Points of Interest
:------------------------:|:-----------------------:
<img src="measure.png" style="width:350px;height:120px;">  |  <img src="wingpose.png" style="width:300px;height:150px;">

## Labeling Steps

### 1. Load Images
Select the folder button at the bottom left to upload a folder of images

### 2. Contour
Necessary for perimeter or area of object
1. Select polygon tool
2. Begin following perimeter with clicks
    - Use enough points to accurately represent shape
3. Press enter when done creating contour

When done with all contours for a file go to file -> save -> COCO JSON
- This only saves the contour coordinates 


<br/><br/><br/><br/><br/><br/>
  <br/>

Contour |
:------------------------:|
<img src="contour.png" style="width:350px;height:150px;"> |

### 3. Points of interest
If a contour was not created draw a rectangle around the object to create points
1. Select points in consistent order for every image
- Your 1st click should always represent the same feature of an object

When done with all points for folder go to file -> save -> Dlib XML
- This only saves the points of interest coordinates

Points |
:------------------------:|
<img src="newpoints.PNG" style="width:650px;height:350px;"> |

If you ever want to continue working on a folder file -> save -> Project file
Then load by
1. Uploading previous image folder
2. Open -> selected saved project file (.nimn)

<br/><br/><br/><br/><br/><br/><br/>