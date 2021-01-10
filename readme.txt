READ.ME
Create panoarma image from 2 images using homography matrix to find objects.
input from command line:
python Panorama.py path_left_img path_right_img path_output
Pipeline:
1. preprocess and SIFT: read images -> gray scale -> same height -> create SIFT object to detect features -> create keypoints and descriptors to both images(left and right).
2. calculate distance to descriptor in one image to another image: implemented by BFMatcher between to descriptors using euclidean distance metric -> best matching procedure  KNN -> validtion best match using technique ration test.
3. calculate homograpy matrix based on keypoints of the images -> Transformation to a common plane.
4. stitching two images to one panoramic image.
5. corp black sections from panoramic image.
output:
path_output.jpg    