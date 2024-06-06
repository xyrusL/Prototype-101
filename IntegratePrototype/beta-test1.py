import csv
import mediapipe as mp

from keypointClassifier import KeyPointClassifier
from drawHandMarks import draw_landmarks
from collections import deque
from boundingBox import *

import pyfirmata
import urllib.request
import urllib.error
from PIL import Image
from io import BytesIO

args = get_args()
cap_width = args.width
cap_height = args.height

board = pyfirmata.Arduino(args.port)
servo_pinX = board.get_pin('d:9:s')  # pin 9 Arduino
servo_pinY = board.get_pin('d:10:s')  # pin 10 Arduino

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

# Function to draw the Terminator-style overlay
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

# Main function
def main():
    global prev_servoX, prev_servoY, servoX, servoY, target_status
    cap_device = args.url

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # subPart load
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    # Read labels
    with open('subPart/handKeypointLabel.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

    # Coordinate history
    history_length = 16
    point_history = deque(maxlen=history_length)

    mode = 0

    while True:
        try:
            # Process Key (ESC: end)
            key = cv.waitKey(10)
            if key == 27:  # ESC
                break
            number, mode = select_mode(key, mode)

            # Fetch image from ESP camera
            response = urllib.request.urlopen(cap_device, timeout=5)  # Set a timeout value (e.g., 5 seconds)
            img = Image.open(BytesIO(response.read()))
            img = np.array(img)
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            img = cv.resize(img, (cap_width, cap_height))

            image = cv.flip(img, 1)
            debug_image = copy.deepcopy(image)

            # Detection implementation
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True

            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Bounding box calculation
                    brect = bound_rect_calc(debug_image, hand_landmarks)

                    # Landmark calculation
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)

                    # Write to the dataset file
                    write_csv(number, mode, pre_processed_landmark_list, pre_processed_point_history_list)

                    # Hand sign classification
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                    if hand_sign_id == 2:  # Point gesture
                        point_history.append(landmark_list[8])
                    else:
                        point_history.append([0, 0])

                    # Drawing part
                    debug_image = bound_rect_create(use_brect, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    debug_image = info_text(debug_image, brect, handedness, keypoint_classifier_labels[hand_sign_id])

                    # Get the bounding box for the hand
                    bbox = cv.boundingRect(
                        np.array([[lm.x * cap_width, lm.y * cap_height] for lm in hand_landmarks.landmark]).astype(int))
                    x, y, w, h = bbox

                    # Calculate center of the bounding box
                    bbox_center_x = x + w // 2
                    bbox_center_y = y + h // 2

                    # Exponential moving average for smooth movement
                    smooth_factor = 0.2
                    target_servo_x = np.interp(bbox_center_x, [0, cap_width], [0, 180])
                    target_servo_y = np.interp(bbox_center_y, [0, cap_height], [0, 180])
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
                # Hand not detected
                point_history.append([0, 0])

                # Write to servo pins
                servo_pinX.write(prev_servoX)
                servo_pinY.write(prev_servoY)

                # Update target_status
                target_status = "Target: No Hand Detected"
                # Draw Terminator-style overlay
                debug_image = draw_terminator_overlay(debug_image, 90, 90, target_status)

            # Show the debug image
            cv.imshow('Hand Tracking', debug_image)

        except urllib.error.URLError as e:
            error_message = f"Error: {e.reason}"
            black_frame = np.zeros((cap_height, cap_width, 3), np.uint8)
            debug_image = draw_terminator_overlay(black_frame, prev_servoX, prev_servoY, error_message)
            cv.imshow('Hand Tracking', debug_image)
            print(f"Error: {e.reason}")
            print(f"\033[92mTrying to reconnect...\033[0m")
            cv.waitKey(1000)
            continue

        except Exception as e:
            error_message = f"Error: {str(e)}"
            black_frame = np.zeros((cap_height, cap_width, 3), np.uint8)
            debug_image = draw_terminator_overlay(black_frame, prev_servoX, prev_servoY, error_message)
            cv.imshow('Hand Tracking', debug_image)
            print(f"Error: {str(e)}")
            print(f"\033[92mTrying to reconnect...\033[0m")
            cv.waitKey(1000)
            continue

    cv.destroyAllWindows()
    board.exit()

# Initialize a dictionary to keep track of key presses
key_presses = {i: 0 for i in range(97, 123)}
def select_mode(key, mode):
    number = -1
    if 97 <= key <= 122:  # a ~ z
        # Check if the key has been pressed less than 100 times
        if key_presses[key] < 100:
            number = key - 97
            key_presses[key] += 1
        else:
            print(f"Key '{chr(key)}' has already been pressed 100 times. Please press another key.")
    if key == 48:  # n
        mode = 0
    if key == 49:  # k
        mode = 1
    if key == 50:  # h
        mode = 2
    return number, mode

# Function to write the data to a CSV file
def write_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass

    if mode == 1 and (0 <= number <= 25):
        # Define the path to the CSV file
        csv_path = 'subPart/handKeypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            # Write the number and landmark list to the CSV file
            writer.writerow([number, *landmark_list])

    if mode == 2 and (0 <= number <= 9):
        # Define the path to the CSV file
        csv_path = 'subPart/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            # Write the number and point history list to the CSV file
            writer.writerow([number, *point_history_list])
    return

# Function to create a bounding rectangle on the image
def bound_rect_create(use_brect, image, brect):
    if use_brect:
        # Draw a rectangle on the image
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)
    return image

# Function to add text information on the image
def info_text(image, brect, handedness, hand_sign_text):
    # Draw a rectangle for the text background
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 28),
                 (0, 0, 0), -1)
    # Prepare the information text
    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text

    # Put the information text on the image
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv.LINE_AA)

    return image

# Function to display mode and number information on the image
def info(image, mode, number):
    # Define the mode strings
    mode_string = ['Logging Key Point', 'Logging Point History']

    if 1 <= mode <= 2:
        # Display the mode on the image
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

        if 0 <= number <= 9:
            # Display the number on the image
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    return image

# Main function
if __name__ == '__main__':
    main()