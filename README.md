## PanoramaImage 📷

## Description

Create panoarma image from 2 images using homography matrix to find objects.

## Required Libraries & Developing environment
* Operation system: Win 10
* Python version: 3.7 and up
* IDE Pycharm 2020.2.3

Libraries:
* sys
* openCV
* imutils
* numpy

## How to run
Input from command line:
python Panorama.py path_left_img path_right_img path_output

## Pipeline
1. Preprocess and SIFT: read images -> gray scale -> same height -> create SIFT object to detect features -> create keypoints and descriptors to both images(left and right).
2. Calculate distance to descriptor in one image to another image: implemented by BFMatcher between to descriptors using euclidean distance metric -> best matching procedure  KNN -> validtion best match using technique ration test.
3. Calculate homograpy matrix based on keypoints of the images -> Transformation to a common plane.
4. Stitching two images to one panoramic image.
5. Crop black sections from panoramic image.
## output
path_output.jpg
