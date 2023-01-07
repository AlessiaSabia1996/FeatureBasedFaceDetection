import numpy as np
import cv2 as cv
from scipy import ndimage


def cannyEdgeDetection(sigma, img):
    # Calcolo automatico soglie per l'algoritmo di edge detection di Canny
    v = np.median(img)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    return cv.Canny(img, threshold1=lower, threshold2=upper, L2gradient=True)


def robertsCrossEdgeDetection(img):
    img_tmp = np.asarray(img, dtype="int32")

    roberts_cross_v = np.array([[1, 0],
                                [0, -1]])

    roberts_cross_h = np.array([[0, 1],
                                [-1, 0]])

    vertical = ndimage.convolve(img_tmp, roberts_cross_v)
    horizontal = ndimage.convolve(img_tmp, roberts_cross_h)

    edged_img = np.sqrt(np.square(horizontal) + np.square(vertical))

    edged_img = np.asarray(np.clip(edged_img, 0, 255), dtype="uint8")

    # Thresholding binario
    _, edges = cv.threshold(edged_img, 20, 255, cv.THRESH_BINARY)

    return edgeThinning(img, edges)


def sobelEdgeDetection(img, kernelSize):
    grad_x = cv.Sobel(img, cv.CV_16S, 1, 0, ksize=kernelSize, scale=1, delta=0, borderType=cv.BORDER_CONSTANT)
    grad_y = cv.Sobel(img, cv.CV_16S, 0, 1, ksize=kernelSize, scale=1, delta=0, borderType=cv.BORDER_CONSTANT)
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)
    # edges è una matrice contenente soli 0, 128 o 256
    edges = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    # Thresholding binario
    retval, edges = cv.threshold(edges, 70, 255, cv.THRESH_BINARY)

    return edgeThinning(img, edges)


def edgeThinning(img, edges):
    # Edge thinning:
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    # Creiamo un'immagine di output vuota per memorizzare i valori
    thin = np.zeros(img.shape, dtype='uint8')

    # Iteriamo finché l'erosione non ci porta a un insieme vuoto
    while cv.countNonZero(edges) != 0:
        # Erosion
        erode = cv.erode(edges, kernel)
        # Opening on eroded image
        opening = cv.morphologyEx(erode, cv.MORPH_OPEN, kernel)
        # Subtract these two
        subset = erode - opening
        # Union of all previous sets
        thin = cv.bitwise_or(subset, thin)
        # Set the eroded image for next iteration
        edges = erode.copy()
    return thin
