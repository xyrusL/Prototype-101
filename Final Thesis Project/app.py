import csv
import mediapipe as mp
import pygame
import time
import threading
from utils import CvFpsCalc
from subPart.keyPointClassifier import KeyPointClassifier
from subPart.args import get_args
from subPart.draw_landmarks import draw_landmarks
from subPart.bounding_rect import calc_bounding_rect
from subPart.process import *

# Distribution or sharing without explicit permission from me is strictly prohibited.
# Note: I, dev.paul, do not grant permission to distribute or share this code with the public or any third party.
# Purpose: This program builds upon and customizes the original code to provide enhanced functionality and improvements.

# Setup parameters
showLandmarks = False  # Toggle to show landmarks on the frame
showDetected = False  # Toggle to show detection information on the frame
soundDetected = True  # Toggle to play sound when hand is detected
maximumHand = 4  # Maximum number of hands to detect
cap_width = 1280  # Capture width
cap_height = 720  # Capture height
min_detection_confidence = 0.8  # Minimum confidence threshold for hand detection
min_tracking_confidence = 0.5  # Minimum confidence threshold for hand tracking
def main():
    # Parse command-line arguments
    args = get_args()
    cap_device = args.device
    use_static_image_mode = args.use_static_image_mode
    use_brect = True  # Toggle to use bounding rectangle visualization

    # Initialize video capture
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Initialize Mediapipe Hands module
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=maximumHand,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # Load machine learning models for hand classification
    right_model = KeyPointClassifier('model/rightHandModel.tflite')
    left_model = KeyPointClassifier('model/leftHandModel.tflite')

    previous_letter = ""

    # Load label data for each hand model
    with open('model/rightHand.csv', encoding='utf-8-sig') as f:
        right_hand_labels = [row[0] for row in csv.reader(f)]
    with open('model/leftHand.csv', encoding='utf-8-sig') as f:
        left_hand_labels = [row[0] for row in csv.reader(f)]

    # Initialize FPS calculation utility
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Main loop to process frames
    while True:
        # Calculate and retrieve FPS
        fps = cvFpsCalc.get()

        # Check for ESC key press to exit
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break

        # Capture frame-by-frame
        ret, image = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for natural viewing
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        # Convert the frame from BGR to RGB color space for Mediapipe
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the frame with Mediapipe Hands
        results = hands.process(image)
        image.flags.writeable = True

        hand_detected = "None"

        # If hands are detected, process each hand
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Calculate bounding rectangle and landmarks
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Preprocess landmarks for classification
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                # Classify hand based on handedness
                if handedness.classification[0].label == 'Right':
                    hand_sign_id = right_model(pre_processed_landmark_list)
                    hand_sign_label = right_hand_labels[hand_sign_id]
                    hand_detected = "Right"

                elif handedness.classification[0].label == 'Left':
                    hand_sign_id = left_model(pre_processed_landmark_list)
                    hand_sign_label = left_hand_labels[hand_sign_id]
                    hand_detected = "Left"

                # Draw bounding rectangle and information on debug image
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_info_text(debug_image, brect, handedness, hand_sign_label, hand_detected,
                                             showDetected)

                # Optionally draw landmarks on debug image
                if showLandmarks:
                    debug_image = draw_landmarks(debug_image, landmark_list)

                if soundDetected:
                    if previous_letter != hand_sign_label:
                        play_sound_for_hand_sign(hand_sign_label)
                        previous_letter = hand_sign_label

        # Draw FPS information on debug image
        debug_image = draw_info(debug_image, fps)

        # Display debug image with annotations
        cv.imshow('FSL DETECTION BY CS SAN MATEO', debug_image)

    # Release video capture and close all OpenCV windows
    cap.release()
    cv.destroyAllWindows()

def play_sound_for_hand_sign(hand_sign_label):
    def play_sound():
        time.sleep(0.3)
        file_path = f'voices/{hand_sign_label}.mp3'
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
    threading.Thread(target=play_sound).start()

if __name__ == '__main__':
    main()
