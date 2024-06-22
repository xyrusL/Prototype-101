import csv
import mediapipe as mp
import pygame
import urllib.request
import numpy as np
import pyfirmata
from PIL import Image
from io import BytesIO
from utils import CvFpsCalc
from subPart.keyPointClassifier import KeyPointClassifier
from subPart.args import get_args
from subPart.draw_landmarks import draw_landmarks
from subPart.bounding_rect import calc_bounding_rect
from subPart.process import *

# Arduino setup
board = pyfirmata.Arduino('COM3')  # Update with your Arduino port
servo_x = board.get_pin('d:9:s')  # Servo connected to pin 9
servo_y = board.get_pin('d:10:s')  # Servo connected to pin 10

showLandMarks = True
soundDetected = True

cap_width = 900
cap_height = 600
min_detection_confidence = 0.8
min_tracking_confidence = 0.5

esp_camera_url = 'http://192.168.254.137/cam-hi.jpg'  # Global variable for ESP camera IP address

# Initialize previous positions and smooth factor
prev_center_x, prev_center_y = cap_width // 2, cap_height // 2
prev_servoX, prev_servoY = 90, 90  # Start with servos centered
smooth_factor = 0.2

def main():
    args = get_args()
    use_static_image_mode = args.use_static_image_mode
    use_brect = True

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

    global prev_center_x, prev_center_y, prev_servoX, prev_servoY

    while True:
        try:
            fps = cvFpsCalc.get()

            key = cv.waitKey(10)
            if key == 27:  # ESC
                break

            response = urllib.request.urlopen(esp_camera_url, timeout=1)
            img = Image.open(BytesIO(response.read()))
            img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
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

                    # Use wrist position (landmark 0)
                    wrist_x, wrist_y = landmark_list[0]

                    move_servo(wrist_x, wrist_y)

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
            else:
                # Use previous center positions if hand not detected
                move_servo(prev_center_x, prev_center_y)

            debug_image = draw_info(debug_image, fps)

            # Draw the crosshair
            draw_crosshair(debug_image, (cap_width // 2, cap_height // 2))

            cv.imshow('FSL RECOGNITION BY CS SAN MATEO - DEV.PAUL', debug_image)

        except Exception as e:
            print(f"Error: {str(e)}")
            print(f"\033[92mTrying to reconnect...\033[0m")
            cv.waitKey(1000)
            continue

    cv.destroyAllWindows()

def move_servo(center_x, center_y):
    global prev_center_x, prev_center_y, prev_servoX, prev_servoY

    # Ensure the wrist position does not go out of the frame
    center_x = np.clip(center_x, 0, cap_width)
    center_y = np.clip(center_y, 0, cap_height)

    # Calculate offsets from the center of the frame
    offset_x = center_x - cap_width // 2
    offset_y = center_y - cap_height // 2

    # Map offsets to servo angles, ensuring full range is covered
    target_servo_x = np.interp(offset_x, [-cap_width // 2, cap_width // 2], [-10, 200])
    target_servo_y = np.interp(offset_y, [-cap_height // 2, cap_height // 2], [-10, 200])

    # Calculate new servo positions with a smooth transition
    servoX = int(smooth_factor * target_servo_x + (1 - smooth_factor) * prev_servoX)
    servoY = int(smooth_factor * target_servo_y + (1 - smooth_factor) * prev_servoY)

    servo_x.write(servoX)
    servo_y.write(servoY)

    # Update previous positions
    prev_center_x, prev_center_y = center_x, center_y
    prev_servoX, prev_servoY = servoX, servoY

def draw_crosshair(image, center):
    x, y = center
    cv.line(image, (x - 10, y), (x + 10, y), (0, 255, 0), 1)
    cv.line(image, (x, y - 10), (x, y + 10), (0, 255, 0), 1)

def play_sound_for_hand_sign(hand_sign_label):
    def play_sound():
        file_path = f'voices/{hand_sign_label}.mp3'
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
    play_sound()

if __name__ == '__main__':
    main()
import csv
import mediapipe as mp
import pygame
import urllib.request
import numpy as np
import pyfirmata
from PIL import Image
from io import BytesIO
from utils import CvFpsCalc
from subPart.keyPointClassifier import KeyPointClassifier
from subPart.args import get_args
from subPart.draw_landmarks import draw_landmarks
from subPart.bounding_rect import calc_bounding_rect
from subPart.process import *

# Arduino setup
board = pyfirmata.Arduino('COM3')  # Update with your Arduino port
servo_x = board.get_pin('d:9:s')  # Servo connected to pin 9
servo_y = board.get_pin('d:10:s')  # Servo connected to pin 10

showLandMarks = True
soundDetected = True

cap_width = 900
cap_height = 600
min_detection_confidence = 0.8
min_tracking_confidence = 0.5

esp_camera_url = 'http://192.168.254.137/cam-hi.jpg'  # Global variable for ESP camera IP address

# Initialize previous positions and smooth factor
prev_center_x, prev_center_y = cap_width // 2, cap_height // 2
prev_servoX, prev_servoY = 90, 90  # Start with servos centered
smooth_factor = 0.2

def main():
    args = get_args()
    use_static_image_mode = args.use_static_image_mode
    use_brect = True

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

    global prev_center_x, prev_center_y, prev_servoX, prev_servoY

    while True:
        try:
            fps = cvFpsCalc.get()

            key = cv.waitKey(10)
            if key == 27:  # ESC
                break

            response = urllib.request.urlopen(esp_camera_url, timeout=1)
            img = Image.open(BytesIO(response.read()))
            img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
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

                    # Use wrist position (landmark 0)
                    wrist_x, wrist_y = landmark_list[0]

                    move_servo(wrist_x, wrist_y)

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
            else:
                # Use previous center positions if hand not detected
                move_servo(prev_center_x, prev_center_y)

            debug_image = draw_info(debug_image, fps)

            # Draw the crosshair
            draw_crosshair(debug_image, (cap_width // 2, cap_height // 2))

            cv.imshow('FSL RECOGNITION BY CS SAN MATEO - DEV.PAUL', debug_image)

        except Exception as e:
            print(f"Error: {str(e)}")
            print(f"\033[92mTrying to reconnect...\033[0m")
            cv.waitKey(1000)
            continue

    cv.destroyAllWindows()

def move_servo(center_x, center_y):
    global prev_center_x, prev_center_y, prev_servoX, prev_servoY

    # Ensure the wrist position does not go out of the frame
    center_x = np.clip(center_x, 0, cap_width)
    center_y = np.clip(center_y, 0, cap_height)

    # Calculate offsets from the center of the frame
    offset_x = center_x - cap_width // 2
    offset_y = center_y - cap_height // 2

    # Map offsets to servo angles, ensuring full range is covered
    target_servo_x = np.interp(offset_x, [-cap_width // 2, cap_width // 2], [-10, 200])
    target_servo_y = np.interp(offset_y, [-cap_height // 2, cap_height // 2], [-10, 200])

    # Calculate new servo positions with a smooth transition
    servoX = int(smooth_factor * target_servo_x + (1 - smooth_factor) * prev_servoX)
    servoY = int(smooth_factor * target_servo_y + (1 - smooth_factor) * prev_servoY)

    servo_x.write(servoX)
    servo_y.write(servoY)

    # Update previous positions
    prev_center_x, prev_center_y = center_x, center_y
    prev_servoX, prev_servoY = servoX, servoY

def draw_crosshair(image, center):
    x, y = center
    cv.line(image, (x - 10, y), (x + 10, y), (0, 255, 0), 1)
    cv.line(image, (x, y - 10), (x, y + 10), (0, 255, 0), 1)

def play_sound_for_hand_sign(hand_sign_label):
    def play_sound():
        file_path = f'voices/{hand_sign_label}.mp3'
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
    play_sound()

if __name__ == '__main__':
    main()
