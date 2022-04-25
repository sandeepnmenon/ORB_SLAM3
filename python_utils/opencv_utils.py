import cv2
import numpy as np


def get_orb_matches(des1, des2):
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)
    # matches = matcher.match(des1,des2)

    # store all the good matches as per Lowe's ratio test.
    good_matches = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good_matches.append(m)
            
    return good_matches
