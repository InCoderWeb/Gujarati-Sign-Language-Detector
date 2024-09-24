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


# import cv2
# from cvzone.ClassificationModule import Classifier
# from cvzone.HandTrackingModule import HandDetector
# import numpy as np
# import math
# import time
#
# cap = cv2.VideoCapture(0)
# detector = HandDetector(maxHands=2)
# classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
# offset = 20
# imgSize = 300
#
# folder = "Data/ka"
# counter = 0
# labels = ["ક", "ખ"]
#
# while True:
#     success, frame = cap.read()
#     if not success:
#         break
#
#     hands, frame = detector.findHands(frame)
#     if hands:
#         hand = hands[0]
#         x, y, w, h = hand['bbox']
#         imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
#
#         # Ensure the bounding box is within the frame dimensions
#         if w > 0 and h > 0 and x >= 0 and y >= 0 and (x + w) <= frame.shape[1] and (y + h) <= frame.shape[0]:
#             imageCrop = frame[y - offset:y + h + offset, x - offset:x + w + offset]
#
#             aspectRatio = h / w
#             if aspectRatio > 1:
#                 k = imgSize / h
#                 wCalc = math.ceil(k * w)
#                 imageResize = cv2.resize(imageCrop, (wCalc, imgSize))
#                 imgResizeShape = imageResize.shape
#                 wGap = math.ceil((imgSize - wCalc) / 2)
#                 imgWhite[:, wGap:wCalc + wGap] = imageResize
#                 prediction, index = classifier.getPrediction(imgWhite, draw=False)
#                 print(prediction, index)
#
#             else:
#                 k = imgSize / w
#                 hCalc = math.ceil(k * h)
#                 imageResize = cv2.resize(imageCrop, (imgSize, hCalc))
#                 imgResizeShape = imageResize.shape
#                 hGap = math.ceil((imgSize - hCalc) / 2)
#                 imgWhite[hGap:hCalc + hGap, :] = imageResize
#
#             cv2.imshow('crop', imageCrop)
#             cv2.imshow('Image White', imgWhite)
#
#     cv2.imshow('frame', frame)
#     cv2.waitKey(1)
#
# cap.release()
# cv2.destroyAllWindows()


# import cv2
# from cvzone.ClassificationModule import Classifier
# from cvzone.HandTrackingModule import HandDetector
# import numpy as np
# import math
# import tensorflow as tf
#
# # Logic to handle TensorFlow/Keras version discrepancies
# from tensorflow.keras import layers
#
#
# class CustomDepthwiseConv2D(layers.DepthwiseConv2D):
#     def __init__(self, kernel_size, strides=(1, 1), padding='valid', depth_multiplier=1, data_format=None,
#                  dilation_rate=(1, 1), activation=None, use_bias=True, depthwise_initializer='glorot_uniform',
#                  bias_initializer='zeros', depthwise_regularizer=None, bias_regularizer=None,
#                  activity_regularizer=None, depthwise_constraint=None, bias_constraint=None, **kwargs):
#         # Filter out unsupported arguments
#         kwargs.pop('groups', None)
#         super(CustomDepthwiseConv2D, self).__init__(kernel_size, strides, padding, depth_multiplier, data_format,
#                                                     dilation_rate, activation, use_bias, depthwise_initializer,
#                                                     bias_initializer, depthwise_regularizer, bias_regularizer,
#                                                     activity_regularizer, depthwise_constraint, bias_constraint,
#                                                     **kwargs)
#
#
# # Replace 'DepthwiseConv2D' with 'CustomDepthwiseConv2D' in loaded model
# tf.keras.utils.get_custom_objects().update({'DepthwiseConv2D': CustomDepthwiseConv2D})
#
# cap = cv2.VideoCapture(0)
# detector = HandDetector(maxHands=2)
# classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
# offset = 20
# imgSize = 300
#
# folder = "Data/ka"
# counter = 0
# labels = ["ક", "ખ"]
#
# while True:
#     success, frame = cap.read()
#     imgOutput = frame.copy()
#     if not success:
#         break
#
#     hands, frame = detector.findHands(frame)
#     if hands:
#         hand = hands[0]
#         x, y, w, h = hand['bbox']
#         imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
#
#         # Ensure the bounding box is within the frame dimensions
#         if w > 0 and h > 0 and x >= 0 and y >= 0 and (x + w) <= frame.shape[1] and (y + h) <= frame.shape[0]:
#             if y - offset >= 0 and y + h + offset <= frame.shape[0] and x - offset >= 0 and x + w + offset <= \
#                     frame.shape[1]:
#                 imageCrop = frame[y - offset:y + h + offset, x - offset:x + w + offset]
#                 if imageCrop.size != 0:
#                     aspectRatio = h / w
#                     if aspectRatio > 1:
#                         k = imgSize / h
#                         wCalc = math.ceil(k * w)
#                         imageResize = cv2.resize(imageCrop, (wCalc, imgSize))
#                         imgResizeShape = imageResize.shape
#                         wGap = math.ceil((imgSize - wCalc) / 2)
#                         imgWhite[:, wGap:wCalc + wGap] = imageResize
#                         prediction, index = classifier.getPrediction(imgWhite, draw=False)
#                         print(prediction, index)
#                     else:
#                         k = imgSize / w
#                         hCalc = math.ceil(k * h)
#                         imageResize = cv2.resize(imageCrop, (imgSize, hCalc))
#                         imgResizeShape = imageResize.shape
#                         hGap = math.ceil((imgSize - hCalc) / 2)
#                         imgWhite[hGap:hCalc + hGap, :] = imageResize
#
#                     cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
#                                   (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
#                     cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255),
#                                 2)
#                     cv2.rectangle(imgOutput, (x - offset, y - offset),
#                                   (x + w + offset, y + h + offset), (255, 0, 255), 4)
#                     cv2.imshow('crop', imageCrop)
#                     cv2.imshow('Image White', imgWhite)
#
#     cv2.imshow('frame', frame)
#     cv2.waitKey(1)
#
# cap.release()
# cv2.destroyAllWindows()


import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras import layers


# Define a custom DepthwiseConv2D layer
class CustomDepthwiseConv2D(layers.DepthwiseConv2D):
    def __init__(self, kernel_size, strides=(1, 1), padding='valid', depth_multiplier=1, data_format=None,
                 dilation_rate=(1, 1), activation=None, use_bias=True, depthwise_initializer='glorot_uniform',
                 bias_initializer='zeros', depthwise_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, depthwise_constraint=None, bias_constraint=None, **kwargs):
        # Filter out unsupported arguments
        kwargs.pop('groups', None)
        super(CustomDepthwiseConv2D, self).__init__(kernel_size, strides, padding, depth_multiplier, data_format,
                                                    dilation_rate, activation, use_bias, depthwise_initializer,
                                                    bias_initializer, depthwise_regularizer, bias_regularizer,
                                                    activity_regularizer, depthwise_constraint, bias_constraint,
                                                    **kwargs)


# Replace 'DepthwiseConv2D' with 'CustomDepthwiseConv2D' in the loaded model
tf.keras.utils.get_custom_objects().update({'DepthwiseConv2D': CustomDepthwiseConv2D})

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300
folder = "Data/C"
counter = 0
labels = ["KA", "KHA"]
gujaratiLabels = {
    "KA": "ક",
    "KHA": "ખ"
}

while True:
    success, img = cap.read()
    if not success:
        break
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Ensure the bounding box is within the frame dimensions
        if x - offset >= 0 and y - offset >= 0 and x + w + offset <= img.shape[1] and y + h + offset <= img.shape[0]:
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            if imgCrop.size != 0:
                aspectRatio = h / w
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    print(gujaratiLabels[labels[index]])
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)

                cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                              (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
                cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset),
                              (x + w + offset, y + h + offset), (255, 0, 255), 4)
                cv2.imshow("ImageCrop", imgCrop)
                cv2.imshow("ImageWhite", imgWhite)
    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
