import cv2 as cv  # OpenCV


def stdHistogramEqualization(img):
    # Histogram equalization: considera il contrasto globale dell'immagine, spesso non e' una buona idea
    return cv.equalizeHist(img)


def adaptiveHistogramEqualization(img, clipLimit, tileGridSize):
    # Histogram equalization 2: adaptive histogram equalization
    clahe_obj = cv.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe_obj.apply(img)
