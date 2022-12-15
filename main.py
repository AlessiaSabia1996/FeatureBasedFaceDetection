import numpy as np
import cv2 as cv  # OpenCV
from matplotlib import pyplot as plt

import os
from os import listdir

# get the path/directory
folder_dir = "C:\\Users\\Utente\\Università di Napoli Federico II\\CARMINE GRIMALDI - Progetto CV\\BioID Face " \
             "Dataset\\BioID-FaceDatabase-V1.2 "
path = "C:/Users/Utente/OneDrive/Desktop/BioID-FaceDatabase-V1.2/"
for images in os.listdir(path):

    # check if the image ends with png
    if images.endswith(".pgm"):
        img = cv.imread(path + images, 0)
        # cv.imshow("foto", img)
        # cv.waitKey(0)

        # Histogram equalization: considera il contrasto globale dell'immagine, spesso non e' una buona idea
        # equ = cv.equalizeHist(img)
        # res = np.hstack((img, equ))  # Stack di immagini fianco a fianco
        # cv.namedWindow("SX: Originale, DX: Equalizzata", cv.WINDOW_NORMAL)
        # cv.imshow('SX: Originale, DX: Equalizzata', res)
        # cv.waitKey(0)

        # Histogram equalization 2: adaptive histogram equalization
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(img)
        # res = np.hstack((img, equ, cl1))  # Stack di immagini fianco a fianco
        # cv.namedWindow("SX: Originale, DX: Equalizzata", cv.WINDOW_NORMAL)
        # cv.imshow('SX: Originale, DX: Equalizzata', res)
        # cv.waitKey(0)

        # Testing OpenCV Median Filter
        # median = cv.medianBlur(equ, 3)
        # res = np.hstack((img, median))
        # cv.namedWindow("SX: Originale, DX: Equalizzata", cv.WINDOW_NORMAL)
        # cv.imshow('SX: Originale, DX: Equalizzata', res)
        # cv.waitKey(0)

        # Testing OpenCV Median Filter 2
        median2 = cv.medianBlur(cl1, 3)
        # res = np.hstack((median, median2))
        # cv.namedWindow("Primo Mediano, Secondo Mediano", cv.WINDOW_NORMAL)
        # cv.imshow('Primo Mediano, Secondo Mediano', res)
        # cv.waitKey(0)

        retval, median2 = cv.threshold(median2, 75, 255, cv.THRESH_BINARY)

        grad_x = cv.Sobel(median2, cv.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
        grad_y = cv.Sobel(median2, cv.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
        abs_grad_x = cv.convertScaleAbs(grad_x)
        abs_grad_y = cv.convertScaleAbs(grad_y)
        grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        cv.imwrite("C:/Users/Utente/OneDrive/Desktop/BioID-FaceDatabase-V1.2/DatasetPreprocessed/" + str(images), grad)
        # cv.namedWindow("Sobel", cv.WINDOW_NORMAL)
        # cv.imshow('Sobel', grad)
        # cv.waitKey(0)
