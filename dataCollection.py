# import cv2
# from cvzone.HandTrackingModule import HandDetector
#
# cap = cv2.VideoCapture(0)
# detector = HandDetector(maxHands=2)
# offset = 20
#
# while True:
#     success, frame = cap.read()
#     hands, frame = detector.findHands(frame)
#     if hands:
#         hand = hands[0]
#         x, y, w, h = hand['bbox']
#         imageCrop = frame[y - offset:y + h + offset, x - offset:x + w + offset]
#         cv2.imshow('crop', imageCrop)
#
#     cv2.imshow('frame', frame)
#     cv2.waitKey(1)


import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
offset = 20
imgSize = 300

folder = "Data/ca"
counter = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    hands, frame = detector.findHands(frame)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Ensure the bounding box is within the frame dimensions
        if w > 0 and h > 0 and x >= 0 and y >= 0 and (x + w) <= frame.shape[1] and (y + h) <= frame.shape[0]:
            imageCrop = frame[y - offset:y + h + offset, x - offset:x + w + offset]

            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCalc = math.ceil(k * w)
                imageResize = cv2.resize(imageCrop, (wCalc, imgSize))
                imgResizeShape = imageResize.shape
                wGap = math.ceil((imgSize - wCalc) / 2)
                imgWhite[:, wGap:wCalc + wGap] = imageResize
            else:
                k = imgSize / w
                hCalc = math.ceil(k * h)
                imageResize = cv2.resize(imageCrop, (imgSize, hCalc))
                imgResizeShape = imageResize.shape
                hGap = math.ceil((imgSize - hCalc) / 2)
                imgWhite[hGap:hCalc + hGap, :] = imageResize

            cv2.imshow('crop', imageCrop)
            cv2.imshow('Image White', imgWhite)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('s'):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.png', imgWhite)
        print(counter)

cap.release()
cv2.destroyAllWindows()
