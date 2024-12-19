import csv
import time
import mediapipe as mp
import threading
from collections import Counter, deque
from utils import CvFpsCalc
from model import KeyPointClassifier, PointHistoryClassifier
from subPart.args import get_args
from subPart.draw_landmarks import draw_landmarks
from subPart.bounding_rect import calc_bounding_rect
from subPart.process import *

logging_flag = False
logging_number = -1

def logging_function(number, mode, landmark_list, point_history_list, num_per_class):
    global logging_flag, logging_number
    logging_flag = True
    for i in range(num_per_class):
        if mode == 1 and (0 <= number <= 9):
            csv_path = 'model/keypoint_classifier/keypoint.csv'
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, *landmark_list])
        if mode == 2 and (0 <= number <= 9):
            csv_path = 'model/point_history_classifier/point_history.csv'
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, *point_history_list])
        print(f"Saving Label {number} - {i + 1}/{num_per_class}")
        time.sleep(0.4)  # 500 milliseconds
    print(f"Done Save {number}")
    logging_flag = False
    logging_number = -1

def main():
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
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
    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]
    with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [row[0] for row in point_history_classifier_labels]

    cvFpsCalc = CvFpsCalc(buffer_len=10)

    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)
    mode = 0
    number_per_class = 20
    num_to_log = -1  # Initialize to -1

    while True:
        fps = cvFpsCalc.get()

        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode, num_to_log = select_mode(key, mode, num_to_log)

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
                pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)

                if num_to_log >= 0 and not logging_flag:
                    logging_thread = threading.Thread(target=logging_function, args=(num_to_log, mode, pre_processed_landmark_list, pre_processed_point_history_list, number_per_class))
                    logging_thread.start()
                    num_to_log = -1

                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(pre_processed_point_history_list)

                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()

                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(debug_image, brect, handedness, keypoint_classifier_labels[hand_sign_id],
                                             point_history_classifier_labels[most_common_fg_id[0][0]])
        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


key_presses = {i: 0 for i in range(48, 58)}

def select_mode(key, mode, num_to_log):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        if key_presses[key] < 100:
            number = key - 48
            key_presses[key] += 1
            if mode == 1:
                num_to_log = number
        else:
            print(f"Key '{chr(key)}' has already been pressed 100 times. Please press another key.")
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode, num_to_log

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image

if __name__ == '__main__':
    main()