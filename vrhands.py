import cv2
import numpy as np
import mediapipe as mp
import time
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def get_hand_landmarks(hand_model, image):
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hand_model.process(image)
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    return results.multi_hand_landmarks


# For webcam input:
cap = cv2.VideoCapture(0)
hand_model = mp.solutions.hands.Hands(min_detection_confidence=0.75, min_tracking_confidence=0.75)
while cap.isOpened():
    # Get next image, flushing the queue
    success, next_image = True, None
    MAX_FLUSH = 4
    for _ in range(MAX_FLUSH):
        success, next_image = cap.read()

    height, width, channels = next_image.shape
    left_image = cv2.cvtColor(next_image[:,:width//2], cv2.COLOR_BGR2RGB)
    right_image = cv2.cvtColor(next_image[:,width//2:], cv2.COLOR_BGR2RGB)

    left_landmarks = get_hand_landmarks(hand_model, left_image)
    right_landmarks = get_hand_landmarks(hand_model, right_image)

    if left_landmarks and right_landmarks:
        mp_drawing.draw_landmarks(
            left_image,
            left_landmarks[0],
            mp_hands.HAND_CONNECTIONS,
        )
        mp_drawing.draw_landmarks(
            left_image,
            right_landmarks[0],
            mp_hands.HAND_CONNECTIONS,
        )
    cv2.imshow('Hand Tracker', left_image)
    cv2.waitKey(5)

cap.release()

