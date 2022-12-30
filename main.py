import os
import re  # regular expressions
from csv import writer
from localDatasetPath import getDatasetPath
from histogramEqualization import *
from imageFilters import *
from edgeDetection import *
from neuralNetworkTraining import *

# Costanti
MIN_BASE = MIN_HEIGHT = 24  # Dimensioni delle sotto-finestre da individuare


def main():
    # Carico il dataset
    dataset = np.loadtxt('feature_files.csv', delimiter=',')

    # Test addestramento rete neurale
    looTrainNeuralNetwork(dataset, epochs=50)
    kfoldTrainNeuralNetwork(dataset, epochs=50)

    # Indice del numero di sotto-finestre individuate in tutte le immagini del dataset
    count = 0

    # Recupero dei path locali al dataset
    alessia_path = getDatasetPath("alessia")
    francesco_path = getDatasetPath("francesco")
    carmine_path = getDatasetPath("carmine")

    # Definizione del path locale al dataset da utilizzare da qui in poi
    global_path = carmine_path

    for images in os.listdir(global_path):
        # check if the image ends with png
        if images.endswith(".pgm"):
            img = cv.imread(global_path + images, 0)
            # cv.imshow("foto", img)
            # cv.waitKey(0)

            # Standard histogram equalization:
            # equ = stdHistogramEqualization(img)
            # res = np.hstack((img, equ))  # Stack di immagini fianco a fianco
            # cv.namedWindow("SX: Originale, DX: Equalizzata", cv.WINDOW_NORMAL)
            # cv.imshow('SX: Originale, DX: Equalizzata', res)
            # cv.waitKey(0)

            # Histogram equalization 2: adaptive histogram equalization
            clahe = adaptiveHistogramEqualization(img, clipLimit=2.0, tileGridSize=(8, 8))
            # res = np.hstack((img, equ, clahe))  # Stack di immagini fianco a fianco
            # cv.namedWindow("SX: Originale, DX: Equalizzata", cv.WINDOW_NORMAL)
            # cv.imshow('SX: Originale, DX: Equalizzata', res)
            # cv.waitKey(0)

            # Test filtro mediano OpenCV
            # median = medianBlur(equ, 3)
            # res = np.hstack((img, median))
            # cv.namedWindow("SX: Originale, DX: Equalizzata", cv.WINDOW_NORMAL)
            # cv.imshow('SX: Originale, DX: Equalizzata', res)
            # cv.waitKey(0)

            # Test filtro Gaussiano OpenCV
            gaussian = gaussianBlur(clahe, (5, 5), cv.BORDER_CONSTANT)
            # res = np.hstack((median, gaussian))
            # cv.namedWindow("Confronto filtro Mediano e Gaussiano", cv.WINDOW_NORMAL)
            # cv.imshow('Confronto filtro Mediano e Gaussiano', res)
            # cv.waitKey(0)

            edges = cannyEdgeDetection(sigma=0.33, img=gaussian)
            # cv.namedWindow("Canny", cv.WINDOW_NORMAL)
            # cv.imshow('Canny', edges)
            # cv.waitKey(0)

            # Caricamento del modello preaddestrato del classificatore di Viola-Jones di OpenCV
            eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')

            # edges = sobelEdgeDetection(img=gaussian, kernelSize=3)
            # cv.namedWindow("Sobel", cv.WINDOW_NORMAL)
            # cv.imshow('Sobel', edges)
            # cv.waitKey(0)

            top_left_coord = ()
            top_right_coord = ()
            bottom_left_coord = ()
            bottom_right_coord = ()

            # Estrapolo le quattro coordinate della finestra di edge:
            for row_index, row in enumerate(edges):
                for col_index, col in enumerate(row):

                    # i. Search for top-left, top-right, bottomleft and bottom-right coordinates.
                    if col != 0:  # Allora questo è un pixel di edge
                        # L'ordine di aggiornamento è: top_left, top_right
                        if not top_left_coord:
                            top_left_coord = (row_index, col_index)
                        elif not top_right_coord and col_index - top_left_coord[1] > MIN_BASE:
                            top_right_coord = (row_index, col_index)

                    if len(top_left_coord) > 0 and len(top_right_coord) > 0:
                        # Allora ho trovato le due coordinate superiori della finestra
                        # ricerco il pixel di edge verso le colonne di queste coordinate

                        for actual_row_index, actual_row in enumerate(edges):
                            if actual_row_index > top_left_coord[0]:
                                if actual_row[top_left_coord[1]] != 0 and \
                                        actual_row_index - top_left_coord[0] > MIN_HEIGHT:
                                    # Allora questo è il pixel di edge in basso a sinistra
                                    bottom_left_coord = (actual_row_index, top_left_coord[1])
                                    bottom_right_coord = (actual_row_index, top_right_coord[1])
                                    break
                                elif actual_row[top_right_coord[1]] != 0 and \
                                        actual_row_index - top_right_coord[0] > MIN_HEIGHT:
                                    # Allora questo è il pixel di edge in basso a destra
                                    bottom_right_coord = (actual_row_index, top_right_coord[1])
                                    bottom_left_coord = (actual_row_index, top_left_coord[1])
                                    break

                        if len(bottom_left_coord) < 1 and len(bottom_right_coord) < 1:
                            # Allora non ho trovato un edge inferiore su entrambe le due colonne,
                            # occorre ripulire le informazioni delle coordinate top
                            top_left_coord = ()
                            top_right_coord = ()
                            bottom_right_coord = ()
                            bottom_left_coord = ()

                        # Se ho individuato una finestra ne calcolo la media
                        elif len(bottom_left_coord) > 0 and len(bottom_right_coord) > 0:
                            edges_found_gray = edges.copy()
                            edges_found = cv.cvtColor(edges_found_gray, cv.COLOR_GRAY2RGB)
                            edges_found = cv.rectangle(edges_found, (top_left_coord[1], top_left_coord[0]),
                                                       (bottom_right_coord[1], bottom_right_coord[0]), (0, 255, 0),
                                                       thickness=1)
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
                                        if bottom_left_coord[1] <= col_index_2 <= \
                                                bottom_right_coord[1]:  # Z <= row_index_2 <= W
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
                                sub_reg_img = gaussian[y:h, x:w]
                                tmp = img.copy()
                                # salvo la sottofinestra
                                # sub_reg_col = tmp[y:h, x:w]
                                # faccio il rilevamento degli occhi
                                eyes = eye_cascade.detectMultiScale(sub_reg_img)
                                # Per un rilevamento con meno falsi positivi provare:
                                # eyes = eye_cascade.detectMultiScale(sub_reg_img, scaleFactor=1.3, minNeighbors=5)

                                if len(eyes) > 0:
                                    with open(global_path + 'eyeFiles/' + images[:len(images) - 4] + '.eye') as f:
                                        coordinates = [int(coord) for coord in re.findall(r'\b\d+\b', f.read())]
                                        # print(coordinates)
                                        # Ricordiamo che eyes è riferito a una sottoregione
                                        # dell'immagine mentre coordinates all'intera immagine (senza neppure filtraggi)

                                    for (ex, ey, ew, eh) in eyes:
                                        # (ex, ey) colonna x riga top left della sotto finestra
                                        # (ex+ew, ey+eh) colonna x riga bottom right della sotto finestra
                                        # print(x + ex, y + ey, x + ex + ew,
                                        #      y + ey + eh)  # x e y mi permettono di ottenere le coordinate
                                        # dall'immagine ritagliata a quella intera

                                        # Se la coordinata del dataset dell'occhio destro è contenuta nel rettangolo
                                        # individuato dalle quattro coordinate allora questo è un occhio e non un falso positivo:

                                        max_value = max(ex + ew, ey + eh)

                                        # Controllo occhio dx
                                        if y + ey <= coordinates[1] <= y + ey + eh and \
                                                x + ex <= coordinates[0] <= x + ex + ew:
                                            print("occhio destro trovato")
                                            # cv.rectangle(sub_reg_col, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)
                                            normalized_coord = [ex / max_value, ey / max_value,
                                                                (ex + ew) / max_value, (ey + eh) / max_value, 1]

                                        # Controllo occhio sx
                                        elif y + ey <= coordinates[3] <= y + ey + eh and \
                                                x + ex <= coordinates[2] <= x + ex + ew:
                                            print("occhio sinistro trovato")
                                            normalized_coord = [ex / max_value, ey / max_value,
                                                                (ex + ew) / max_value, (ey + eh) / max_value, 1]
                                        else:
                                            print("La seguente non è una face region")
                                            normalized_coord = [ex / max_value, ey / max_value,
                                                                (ex + ew) / max_value, (ey + eh) / max_value, 0]

                                        # cv.rectangle(sub_reg_col, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)
                                        with open('feature_files.csv', 'a+', newline='') as f:
                                            writer_obj = writer(f)
                                            writer_obj.writerow(normalized_coord)
                                            f.close()

                                        normalized_coord.clear()

                                    cv.rectangle(tmp, (x, y), (w, h), (255, 0, 0), 2)
                                    # sub_reg_col = img.copy()
                                    # sub_reg_col = sub_reg_col[y:h, x:w]
                                    # for (ex, ey, ew, eh) in eyes:
                                    #     cv.rectangle(sub_reg_col, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)

                                    cv.namedWindow("Face", cv.WINDOW_NORMAL)
                                    cv.imshow('Face', tmp)
                                    cv.waitKey(0)
                                else:
                                    print("La seguente non è una face region")
                            else:
                                print("Media = 0")

                            # Ripulisco le coordinate
                            top_left_coord = ()
                            top_right_coord = ()
                            bottom_left_coord = ()
                            bottom_right_coord = ()

                # Ripulisco le coordinate una volta esaurite le colonne della riga
                top_left_coord = ()
                top_right_coord = ()
                bottom_left_coord = ()
                bottom_right_coord = ()

    print("Il numero di sotto-finestre totali individuate e': " + str(count))


if __name__ == "__main__":
    main()
