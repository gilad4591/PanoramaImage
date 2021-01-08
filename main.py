import cv2
import matplotlib.pyplot as plt
import numpy as np


def read_image(path):
    return cv2.imread(path)


def resize(img, w, h):
    return cv2.resize(img, (w, h))


def main():
    image_left = read_image('2/left.jpg')
    image_right = read_image('2/right.jpg')

    image_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB)
    image_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2RGB)

    #resizing and get shape of 2 images
    #left_w, left_h, left_c = image_left.shape
    right_w, right_h, right_c = image_right.shape
    image_left = resize(image_left, right_w, right_h)
    left_w, left_h, left_c = image_left.shape

    #turn the images to grayscale
    image_left_gray = cv2.cvtColor(image_left, cv2.COLOR_RGB2GRAY)
    image_right_gray = cv2.cvtColor(image_right, cv2.COLOR_RGB2GRAY)


    #get keypoints of each image
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints_left, descriptors_left = sift.detectAndCompute(image_left_gray, None)
    keypoints_right, descriptors_right = sift.detectAndCompute(image_right_gray, None)

    #create a matcher - check if point of first image matches to other point and add them to array
    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(descriptors_left, descriptors_right, k=2)
    matches = []
    ratio = 0.85
    for m1, m2 in raw_matches:
        if m1.distance < ratio * m2.distance:
            matches.append(m1)
    if len(matches) >= 4:
        #if theres more than 4 matches it will combine them finally
        left_pts = np.float32([keypoints_left[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        right_pts = np.float32([keypoints_right[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    #create a homography mask
    H, status = cv2.findHomography(right_pts, left_pts, cv2.RANSAC, 5.0)

    #create width and height to panorama image
    width_panorama = left_w + right_h
    height_panorama = right_h

    #fill the first part of the panorama image with the right image, and the left part with left_image
    res = cv2.warpPerspective(image_right, H, (width_panorama, height_panorama))
    res[0:image_left.shape[0], 0:image_left.shape[1]] = image_left

    #get rid of the black threshold with contours
    grayColorImage = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(grayColorImage, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = res[y:y + h, x:x + w]

    #show the image and write to memory
    plt.imshow(crop)
    plt.show()
    crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    cv2.imwrite('panorama.jpg', crop)

if __name__ == "__main__":
    main()
