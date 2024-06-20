import csv
import mediapipe as mp
import pygame
from utils import CvFpsCalc
from subPart.keyPointClassifier import KeyPointClassifier
from subPart.args import get_args
from subPart.draw_landmarks import draw_landmarks
from subPart.bounding_rect import calc_bounding_rect
from subPart.process import *

import numpy as np
import pyfirmata
import urllib.request
import urllib.error
from PIL import Image
from io import BytesIO

# Copyright 2024 PAUL
# This code is a modified version of the original code, adapted for the purpose of our thesis.
# This code is proprietary and confidential.
# It is not to be distributed, modified, or used for commercial purposes without the explicit permission of the author.
# Any unauthorized use, distribution, or modification of this code may result in legal action.

# configuration details
url_ip = "Enter the IP address here"
arduino_port = "COM3"
cap_width = 1280
cap_height = 720

board = pyfirmata.Arduino(arduino_port)
servo_pinX = board.get_pin('d:9:s')  # pin 9 Arduino
servo_pinY = board.get_pin('d:10:s')  # pin 10 Arduino

showLandMarks = False
soundDetected = True

min_detection_confidence = 0.7
min_tracking_confidence = 0.5

# Move servos to the middle position at the start
servo_pinX.write(90)
servo_pinY.write(90)

center_x = cap_width // 2
center_y = cap_height // 2

# Initialize servoX and servoY
servoX = 90
servoY = 90

# Initialize previous_servoX and previous_servoY
prev_servoX = 90
prev_servoY = 90

# Initialize target_status
target_status = "Target: No Hand Detected"

def main():
    global prev_servoX, prev_servoY, servoX, servoY, target_status

    args = get_args()
    cap_device = url_ip
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
        try:
            fps = cvFpsCalc.get()

            key = cv.waitKey(10)
            if key == 27:  # ESC
                break

            # Fetch image from ESP camera
            response = urllib.request.urlopen(cap_device, timeout=5)  # Set a timeout value (e.g., 5 seconds)
            img = Image.open(BytesIO(response.read()))
            img = np.array(img)
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            img = cv.resize(img, (cap_width, cap_height))

            image = cv.flip(img, 1)
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

                    # Get the bounding box for the hand
                    bbox = cv.boundingRect(
                        np.array([[lm.x * cap_width, lm.y * cap_height] for lm in hand_landmarks.landmark]).astype(int))
                    x, y, w, h = bbox

                    # Calculate center of the bounding box
                    bbox_center_x = x + w // 2
                    bbox_center_y = y + h // 2

                    # Define the buffer zone around the edges of the frame
                    buffer_size = 50  # Adjust this value to change the buffer size
                    buffer_x_min = buffer_size
                    buffer_x_max = cap_width - buffer_size
                    buffer_y_min = buffer_size
                    buffer_y_max = cap_height - buffer_size

                    # Check if the bounding box is within the buffer zone
                    if x < buffer_x_min or x + w > buffer_x_max or y < buffer_y_min or y + h > buffer_y_max:
                        # The bounding box is approaching the edge, adjust the servo angles
                        target_servo_x = np.interp(bbox_center_x, [buffer_x_min, buffer_x_max], [0, 180])
                        target_servo_y = np.interp(bbox_center_y, [buffer_y_min, buffer_y_max], [0, 180])
                    else:
                        # The bounding box is within the safe zone, no adjustment needed
                        target_servo_x = np.interp(bbox_center_x, [0, cap_width], [0, 180])
                        target_servo_y = np.interp(bbox_center_y, [0, cap_height], [0, 180])

                    # Exponential moving average for smooth movement
                    smooth_factor = 0.2
                    servoX = int(smooth_factor * target_servo_x + (1 - smooth_factor) * prev_servoX)
                    servoY = int(smooth_factor * target_servo_y + (1 - smooth_factor) * prev_servoY)
                    prev_servoX, prev_servoY = servoX, servoY

                    # Write to servo pins
                    servo_pinX.write(servoX)
                    servo_pinY.write(servoY)

                    # Update target_status
                    target_status = "Target: Hand Detected"

                    # Draw Terminator-style overlay
                    debug_image = draw_terminator_overlay(debug_image, servoX, servoY, target_status)
            else:
                # Write to servo pins
                servo_pinX.write(prev_servoX)
                servo_pinY.write(prev_servoY)

                # Update target_status
                target_status = "Target: No Hand Detected"

                # Draw Terminator-style overlay
                debug_image = draw_terminator_overlay(debug_image, prev_servoX, prev_servoY, target_status)

        except urllib.error.URLError as e:
            error_message = f"Error: {e.reason}"
            black_frame = np.zeros((cap_height, cap_width, 3), np.uint8)
            debug_image = draw_terminator_overlay(black_frame, prev_servoX, prev_servoY, error_message)
            cv.imshow('FSL RECOGNITION BY CS SAN MATEO - DEV.PAUL', debug_image)
            print(f"Error: {e.reason}")
            print(f"\033[92mTrying to reconnect...\033[0m")
            cv.waitKey(1000)
            continue

        except Exception as e:
            error_message = f"Error: {str(e)}"
            black_frame = np.zeros((cap_height, cap_width, 3), np.uint8)
            debug_image = draw_terminator_overlay(black_frame, prev_servoX, prev_servoY, error_message)
            cv.imshow('FSL RECOGNITION BY CS SAN MATEO - DEV.PAUL', debug_image)
            print(f"Error: {str(e)}")
            print(f"\033[92mTrying to reconnect...\033[0m")
            cv.waitKey(1000)
            continue

    cap.release()
    cv.destroyAllWindows()

def play_sound_for_hand_sign(hand_sign_label):
    def play_sound():
        file_path = f'voices/{hand_sign_label}.mp3'
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
    play_sound()

def draw_terminator_overlay(img, servoX, servoY, target_status, error_message=None):
    # Draw information boxes with values
    cv.putText(img, f'Servo X: {servoX} deg', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv.putText(img, f'Servo Y: {servoY} deg', (50, 100), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Draw crosshair at the center
    cv.line(img, (center_x - 20, center_y), (center_x + 20, center_y), (0, 0, 255), 2)
    cv.line(img, (center_x, center_y - 20), (center_x, center_y + 20), (0, 0, 255), 2)

    # Display error message if any
    if error_message:
        cv.rectangle(img, (40, cap_height - 100), (450, cap_height - 50), (0, 0, 0), cv.FILLED)
        cv.putText(img, error_message, (50, cap_height - 70), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv.putText(img, target_status, (50, cap_height - 70), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return img

if __name__ == '__main__':
    main()