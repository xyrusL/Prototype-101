import csv
import mediapipe as mp
import pygame
from utils import CvFpsCalc
from subPart.keyPointClassifier import KeyPointClassifier
from subPart.args import get_args
from subPart.draw_landmarks import draw_landmarks
from subPart.bounding_rect import calc_bounding_rect
from subPart.process import *

# Copyright 2024 PAUL
# This code is a modified version of the original code, adapted for the purpose of our thesis.
# This code is proprietary and confidential.
# It is not to be distributed, modified, or used for commercial purposes without the explicit permission of the author.
# Any unauthorized use, distribution, or modification of this code may result in legal action.

showLandMarks = False
soundDetected = True

cap_width = 1280
cap_height = 720
min_detection_confidence = 0.7
min_tracking_confidence = 0.5

def main():
    args = get_args()
    cap_device = args.device
    use_static_image_mode = args.use_static_image_mode
    use_brect = True

    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    right_hand_model_path = 'Model/rightHandModel.tflite'
    left_hand_model_path = 'Model/leftHandModel.tflite'

    right_hand_classifier = KeyPointClassifier(model_path=right_hand_model_path)
    left_hand_classifier = KeyPointClassifier(model_path=left_hand_model_path)

    with open('Model/rightHand.csv', encoding='utf-8-sig') as f:
        right_hand_labels = csv.reader(f)
        right_hand_labels = [row[0] for row in right_hand_labels]
    with open('Model/leftHand.csv', encoding='utf-8-sig') as f:
        left_hand_labels = csv.reader(f)
        left_hand_labels = [row[0] for row in left_hand_labels]

    cvFpsCalc = CvFpsCalc(buffer_len=10)

    previous_label = ""

    while True:
        fps = cvFpsCalc.get()

        key = cv.waitKey(10)
        if key == 27:  # ESC
            break

        ret, image = cap.read()
        if not ret:
            break

        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                if handedness.classification[0].label == 'Right':
                    hand_sign_id = right_hand_classifier(pre_processed_landmark_list)
                    label = right_hand_labels[hand_sign_id]
                else:
                    hand_sign_id = left_hand_classifier(pre_processed_landmark_list)
                    label = left_hand_labels[hand_sign_id]

                debug_image = draw_info_text(debug_image, brect, handedness, label, label)
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)

                if showLandMarks:
                    debug_image = draw_landmarks(debug_image, landmark_list)

                if soundDetected:
                    if label != previous_label:
                        play_sound_for_hand_sign(label)
                        previous_label = label

        debug_image = draw_info(debug_image, fps)
        cv.imshow('FSL RECOGNITION BY CS SAN MATEO - DEV.PAUL', debug_image)

    cap.release()
    cv.destroyAllWindows()

def play_sound_for_hand_sign(hand_sign_label):
    def play_sound():
        file_path = f'voices2/{hand_sign_label}.mp3'
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
    play_sound()

if __name__ == '__main__':
    main()