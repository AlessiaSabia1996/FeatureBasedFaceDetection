import cv2 as cv  # OpenCV


def medianBlur(img, kernelSize):
    return cv.medianBlur(img, kernelSize)


def gaussianBlur(img, kernelSize, borderConstant):
    return cv.GaussianBlur(img, kernelSize, borderConstant)
