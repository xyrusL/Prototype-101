import urllib.request
import cv2
import csv
import mediapipe as mp

from keypointClassifier import KeyPointClassifier
from drawHandMarks import draw_landmarks
from collections import deque
from boundingBox import *

def main():
    # Argument parsing
    args = get_args()

    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Open video stream from IP camera
    ip_address = 'http://192.168.254.137/1280x720.jpg'

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
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

    # Coordinate history
    history_length = 16
    point_history = deque(maxlen=history_length)

    mode = 0

    while True:
        # Process Key (ESC: end)
        key = cv2.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Open the URL and read the JPEG image
        imgResp = urllib.request.urlopen(ip_address)
        imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
        img = cv2.imdecode(imgNp, cv2.IMREAD_COLOR)

        # Flip the image horizontally
        img = cv2.flip(img, 1)

        image = cv2.resize(img, (cap_width, cap_height))
        debug_image = copy.deepcopy(image)

        # Detection implementation
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
        else:
            point_history.append([0, 0])

        debug_image = info(debug_image, mode, number)

        # Screen reflection
        cv2.imshow('Hand Gesture Recognition', debug_image)

    cv2.destroyAllWindows()

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