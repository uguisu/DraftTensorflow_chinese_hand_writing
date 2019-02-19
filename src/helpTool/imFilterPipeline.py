# encoding: UTF-8

import cv2
import numpy as np


class ImFilterPipeline:

    def __init__(self):
        # init pipeline
        self._pipeline = {
            "rotated": 0,
            "blur": 0,
            "gaussianBlur": 0,
            "resize": 0
        }

    @property
    def pipeline(self):
        return self._pipeline

    def _rotate_bound_with_white_background(self, image, angle):
        """
        Copy from imutils.rotate_bound. Change  background color from (0,0,0) to (255,255,255)
        :param image: image
        :param angle: angle from 0 ~ 360
        :return: processed image
        """
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w / 2, h / 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255))

    def _blur(self, image, ksize=(5, 5)):
        return cv2.blur(image, ksize)

    def _gaussianBlur(self, image, ksize=(5, 5), sigmaX=0):
        return cv2.GaussianBlur(image, ksize, sigmaX)

    def _resize(self, image, target_size=64):
        return cv2.resize(image, (target_size, target_size))

    def filter(self, image):

        rtn = image

        if self._pipeline["rotated"] == 1:
            rtn = self._rotate_bound_with_white_background(rtn, np.random.choice(np.arange(0, 360), 1))
        if self._pipeline["blur"] == 1:
            rtn = self._blur(rtn)
        if self._pipeline["gaussianBlur"] == 1:
            rtn = self._gaussianBlur(rtn)
        if self._pipeline["resize"] == 1:
            rtn = self._resize(rtn)

        return rtn

