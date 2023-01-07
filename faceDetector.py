import tensorflow as tf
import cv2 as cv
import numpy as np
import time
from histogramEqualization import adaptiveHistogramEqualization
from imageFilters import gaussianBlur
from edgeDetection import cannyEdgeDetection
from PIL import Image as PILImage, ImageFont, ImageDraw, ImageOps

SCALE_FACTOR = 1.5
MIN_BASE = int(140 / SCALE_FACTOR)
MIN_HEIGHT = int(180 / SCALE_FACTOR)


def faceDetector():
    model = tf.keras.models.load_model("neuralNetworksModels/modelloReteNeurale-LeaveOneOut.h5")
    vid = cv.VideoCapture(0)
    exitFlag = False
    imageIndex = 1
    rowIndexFlag = 0

    while not exitFlag:
        ret, frame = vid.read()
        print("Ho letto il frame " + str(imageIndex))
        imageIndex = imageIndex+1

        height = int(frame.shape[1])
        width = int(frame.shape[0])

        frame = cv.resize(frame, (int(frame.shape[1] / SCALE_FACTOR), int(frame.shape[0] / SCALE_FACTOR)))

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        clahe = adaptiveHistogramEqualization(gray, clipLimit=2.0, tileGridSize=(8, 8))
        gaussian = gaussianBlur(clahe, (5, 5), cv.BORDER_CONSTANT)
        edges = cannyEdgeDetection(sigma=0.33, img=gaussian)

        top_left_coord = ()
        top_right_coord = ()
        bottom_left_coord = ()
        bottom_right_coord = ()
        goToNextImage = False

        # Estrapolo le quattro coordinate della finestra di edge:
        for row_index, row in enumerate(edges):
            # Faccio predizioni ogni 4 righe
            if rowIndexFlag < 4:
                rowIndexFlag = rowIndexFlag+1
                continue
            else:
                rowIndexFlag = 0

            if goToNextImage is True:
                break

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
                        mean = sum(tmp_list) / len(tmp_list)

                        if mean > 3:  # mean != 0:
                            # print("Media != 0")
                            max_value = max(top_left_coord[1], top_left_coord[0],
                                            bottom_right_coord[1], bottom_right_coord[0])

                            normalized_coord = [[top_left_coord[1] / max_value,
                                                 top_left_coord[0] / max_value,
                                                 bottom_right_coord[1] / max_value,
                                                 bottom_right_coord[0] / max_value]]

                            predictions = model.predict(normalized_coord, verbose=0)
                            # print("Risultato predizione: " + str(predictions[0, 0]))

                            if predictions[0, 0] > 0.85:
                                frame_copy = frame.copy()
                                goToNextImage = True
                                frame_copy = cv.rectangle(frame_copy, (top_left_coord[1], top_left_coord[0]),
                                                          (bottom_right_coord[1], bottom_right_coord[0]),
                                                          (0, 255, 0),
                                                          thickness=1)

                                displayImgWithMsg(frame_copy, "Risultato predizione: " + str(predictions[0, 0]), height, width)
                                # time.sleep(0.2)

                                if cv.waitKey(1) & 0xFF == ord('q'):
                                    exitFlag = True

                            normalized_coord.clear()

                            if goToNextImage is True:
                                break

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

        if goToNextImage is False:
            displayImgWithMsg(frame.copy(), "Risultato predizione: Nessun volto trovato", height, width)
            # time.sleep(2)

            if cv.waitKey(1) & 0xFF == ord('q'):
                exitFlag = True

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv.destroyAllWindows()


def displayImgWithMsg(image, msg, h, w):
    img = ImageOps.expand(PILImage.fromarray(image), border=45, fill=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 36)

    draw.text((0, 0), msg, (0, 0, 0), font=font)

    cv.namedWindow("frame", cv.WINDOW_NORMAL)
    cv.resizeWindow('frame', h, w)
    cv.imshow('frame', np.uint8(img))
