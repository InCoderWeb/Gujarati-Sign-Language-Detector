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
labels = ["KA", "KHA", "GA", "GHA", "CA"]
gujaratiLabels = {
    "KA": "ક",
    "KHA": "ખ",
    "GA": "ગ",
    "GHA": "ઘ",
    "CA": "ચ",
}

# Load Gujarati character images as a dictionary
character_images = {
    "KA": cv2.imread('Char/ka.png', cv2.IMREAD_UNCHANGED),  # Load images with alpha channel (transparency)
    "KHA": cv2.imread('Char/kha.png', cv2.IMREAD_UNCHANGED),
    "GA": cv2.imread('Char/ga.png', cv2.IMREAD_UNCHANGED),
    "GHA": cv2.imread('Char/gha.png', cv2.IMREAD_UNCHANGED),
    "CA": cv2.imread('Char/ca.png', cv2.IMREAD_UNCHANGED),
}


def overlay_image(background, overlay, position, scale=0.5):
    x, y = position

    # Apply the scaling factor to resize the overlay
    overlay_h, overlay_w = overlay.shape[:2]
    new_overlay_w = int(overlay_w * scale)
    new_overlay_h = int(overlay_h * scale)
    overlay = cv2.resize(overlay, (new_overlay_w, new_overlay_h))
    overlay_h, overlay_w = overlay.shape[:2]

    # Ensure the x and y coordinates plus the overlay dimensions fit inside the background
    if y + overlay_h > background.shape[0]:
        overlay_h = background.shape[0] - y
    if x + overlay_w > background.shape[1]:
        overlay_w = background.shape[1] - x
    if overlay_h <= 0 or overlay_w <= 0:
        return  # Do not attempt to overlay if dimensions are invalid

    # Extract region of interest (ROI) from the background image where we will overlay
    roi = background[y:y + overlay_h, x:x + overlay_w]

    # Create masks using the alpha channel
    if overlay.shape[2] == 4:  # Check if overlay has an alpha channel
        overlay_image_gray = overlay[:, :, 3]  # Alpha channel
        mask = overlay_image_gray / 255.0
        inverse_mask = 1 - mask
    else:
        raise ValueError("Overlay image must have an alpha channel")

    # Blend the overlay image with the ROI
    for c in range(3):
        background[y:y + overlay_h, x:x + overlay_w, c] = (
                roi[:, :, c] * inverse_mask + overlay[:, :, c] * mask
        )



# Assuming other parts of your script remain the same, add test images and placeholder functions if needed

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
                if (aspectRatio > 1):
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    predicted_label = labels[index]
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    predicted_label = labels[index]

                # Display the Gujarati character image instead of text
                character_img = character_images[predicted_label]

                # Resize the character image if needed and overlay it at the desired position
                overlay_image(imgOutput, character_img, (x - offset, y - offset - 100))

                # Draw a rectangle around the hand and display it
                cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)
                cv2.imshow("ImageCrop", imgCrop)
                cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
