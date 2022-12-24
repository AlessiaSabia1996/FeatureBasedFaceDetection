import numpy as np
import cv2 as cv  # OpenCV
import os

# get the path/directory
path_Alessia = "C:/Users/Utente/OneDrive/Desktop/BioID-FaceDatabase-V1.2/"
path_Carmine = "C:/Users/Carmine Grimaldi/Desktop/CV Dataset/"
count = 0
for images in os.listdir(path_Carmine):

    # check if the image ends with png
    if images.endswith(".pgm"):
        img = cv.imread(path_Carmine + "BioID_0321.pgm", 0)
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
        median2 = cv.GaussianBlur(cl1, (5, 5), cv.BORDER_CONSTANT)
        # median2 = cv.medianBlur(median2, 5)
        # res = np.hstack((median, median2))
        # cv.namedWindow("Test media + mediana", cv.WINDOW_NORMAL)
        # cv.imshow('Test media + mediana', median2)
        # cv.waitKey(0)

        # ---- Apply automatic Canny edge detection using the computed median----
        sigma = 0.33
        v = np.median(median2)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))

        edges = cv.Canny(median2, threshold1=lower, threshold2=upper, L2gradient=True)
        # cv.namedWindow("Canny", cv.WINDOW_NORMAL)
        # cv.imshow('Canny', edges)
        # cv.waitKey(0)

        # usiamo il modello preaddestrato di Viola Jones che sta nella libreria
        eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')

        ''' 
        grad_x = cv.Sobel(median2, cv.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
        grad_y = cv.Sobel(median2, cv.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
        abs_grad_x = cv.convertScaleAbs(grad_x)
        abs_grad_y = cv.convertScaleAbs(grad_y)
        # edges è una matrice contenente soli 0, 128 o 256
        edges = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        retval, edges = cv.threshold(edges, 70, 255, cv.THRESH_BINARY)

        # Edge thinning:
        # Structuring Element
        kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
        # Create an empty output image to hold values
        thin = np.zeros(img.shape, dtype='uint8')

        # Loop until erosion leads to an empty set
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
        edges = thin
        '''
        # cv.imwrite("C:/Users/Utente/OneDrive/Desktop/BioID-FaceDatabase-V1.2/DatasetPreprocessed/" + str(images), grad)
        # cv.namedWindow("Sobel", cv.WINDOW_NORMAL)
        # cv.imshow('Sobel', edges)
        # cv.waitKey(0)

        top_left_coord = []
        top_right_coord = []
        bottom_left_coord = []
        bottom_right_coord = []

        # Estrapolo le quattro coordinate della finestra di edge:
        for row_index, row in enumerate(edges):
            for col_index, col in enumerate(row):

                # i. Search for top-left, top-right, bottomleft and bottom-right coordinates.
                if col != 0:  # Allora questo è un pixel di edge
                    # L'ordine di aggiornamento è: top_left, top_right
                    if not top_left_coord:
                        top_left_coord.append(row_index)
                        top_left_coord.append(col_index)
                    elif not top_right_coord and col_index - top_left_coord[1] > 5:
                        top_right_coord.append(row_index)
                        top_right_coord.append(col_index)

                if len(top_left_coord) > 0 and len(top_right_coord) > 0:
                    # Allora ho trovato le due coordinate superiori della finestra
                    # ricerco il pixel di edge verso le colonne di queste coordinate
                    column_left = [row_tmp[top_left_coord[1]] for row_tmp in edges]  # top_left_coord[1] è l'indice di colonna
                    column_right = [row_tmp[top_right_coord[1]] for row_tmp in edges]

                    minSize = 24 * 24

                    for actual_row_index, actual_row in enumerate(edges):
                        if actual_row_index > top_left_coord[0]:
                            rowSize = actual_row_index - top_left_coord[0]
                            colSize = top_right_coord[1] - top_left_coord[1]

                            if rowSize * colSize >= minSize:
                                if actual_row[top_left_coord[1]] != 0:
                                    # Allora questo è il pixel di edge in basso a sinistra
                                    bottom_left_coord.append(actual_row_index)
                                    bottom_left_coord.append(top_left_coord[1])
                                    bottom_right_coord.append(actual_row_index)
                                    bottom_right_coord.append(top_right_coord[1])
                                    break
                                elif actual_row[top_right_coord[1]] != 0:
                                    # Allora questo è il pixel di edge in basso a destra
                                    bottom_right_coord.append(actual_row_index)
                                    bottom_right_coord.append(top_right_coord[1])
                                    bottom_left_coord.append(actual_row_index)
                                    bottom_left_coord.append(top_left_coord[1])
                                    break

                    if len(bottom_left_coord) < 1 and len(bottom_right_coord) < 1:
                        # Allora non ho trovato un edge inferiore su entrambe le due colonne,
                        # occorre ripulire le informazioni delle coordinate top
                        top_left_coord.clear()
                        top_right_coord.clear()
                        bottom_right_coord.clear()
                        bottom_left_coord.clear()

                    # Se ho individuato una finestra ne calcolo la media
                    elif len(bottom_left_coord) > 0 and len(bottom_right_coord) > 0:

                        base = abs(top_right_coord[1] - top_left_coord[1])
                        altezza = abs(bottom_left_coord[0] - top_left_coord[0])

                        area = base * altezza
                        # if area > 1000:
                        edges_found_gray = edges.copy()
                        edges_found = cv.cvtColor(edges_found_gray, cv.COLOR_GRAY2RGB)
                        edges_found = cv.rectangle(edges_found, (top_left_coord[1], top_left_coord[0]), (bottom_right_coord[1], bottom_right_coord[0]), (0, 255, 0), thickness=1)
                        cv.namedWindow("Faces", cv.WINDOW_NORMAL)
                        cv.imshow('Faces', edges_found)
                        cv.waitKey(0)

                        # ii. Extract the sub-window from the edge image
                        # X-----------Y
                        # |-----------|
                        # Z-----------W
                        tmp_list = []
                        for row_index_2, row_2 in enumerate(edges):
                            if top_left_coord[0] <= row_index_2 <= bottom_left_coord[0]:  # X <= row_index_2 <= Z
                                for col_index_2, col_2 in enumerate(row_2):
                                    if bottom_left_coord[1] <= col_index_2 <= bottom_right_coord[1]:  # Z <= row_index_2 <= W
                                        tmp_list.append(col_2)
                                    if col_index_2 > bottom_right_coord[1]:
                                        break
                            elif row_index_2 > bottom_left_coord[0]:
                                break

                        # iii. Calculate its mean (µ)
                        count = count + 1
                        mean = sum(tmp_list) / len(tmp_list)

                        if mean > 3:  # mean != 0:
                            print("Media != 0")
                            x = top_left_coord[1]
                            y = top_left_coord[0]
                            w = bottom_right_coord[1]
                            h = bottom_right_coord[0]
                            # sub_reg_img contiene il sottorettangolo che abbiamo trovato prima ma in grayscale
                            sub_reg_img = median2[y:h, x:w]
                            tmp = img.copy()
                            # salvo la sottofinestra
                            sub_reg_col = tmp[y:h, x:w]
                            # faccio il rilevamento degli occhi
                            eyes = eye_cascade.detectMultiScale(sub_reg_img)
                            if len(eyes) > 0:
                                cv.rectangle(tmp, (x, y), (w, h), (255, 0, 0), 2)
                                # sub_reg_col = img.copy()
                                # sub_reg_col = sub_reg_col[y:h, x:w]
                                # for (ex, ey, ew, eh) in eyes:
                                # cv.rectangle(sub_reg_col, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)
                                cv.namedWindow("Face", cv.WINDOW_NORMAL)
                                cv.imshow('Face', tmp)
                                cv.waitKey(0)

                        else:
                            print("Media = 0")

                        # Ripulisco le liste
                        top_left_coord.clear()
                        top_right_coord.clear()
                        bottom_left_coord.clear()
                        bottom_right_coord.clear()

            # Ripulisco le liste delle coordinate una volta esaurite le colonne della riga
            top_left_coord.clear()
            top_right_coord.clear()
            bottom_left_coord.clear()
            bottom_right_coord.clear()
print(count)
