import cv2 as cv
import numpy as np
from PIL import Image as PILImage, ImageFont, ImageDraw, ImageOps


def faceDetector():
    # 0 è l'id della webcam
    vid = cv.VideoCapture(0)
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = vid.read()
        height = int(frame.shape[1])
        width = int(frame.shape[0])

        frame = cv.resize(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)))

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                frame = cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            displayImgWithMsg(frame, "Risultato predizione: True", height, width)
        else:
            displayImgWithMsg(frame, "Risultato predizione: False", height, width)

        # ord('q') - > se premo q oppure dopo 1 millisecondo esco
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

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
