import sys
import csv
import time
import mediapipe as mp
from threading import Thread
import pyfirmata
import requests
import threading
import numpy as np
from queue import Queue
from collections import Counter, deque
from subPart.process import *

# Assuming these are custom modules you've created
from utils import CvFpsCalc
from model import KeyPointClassifier, PointHistoryClassifier
from subPart.args import get_args
from subPart.draw_landmarks import draw_landmarks
from subPart.bounding_rect import calc_bounding_rect
from subPart.process import calc_landmark_list, pre_process_landmark, pre_process_point_history

# Arduino setup
try:
    board = pyfirmata.Arduino('COM3')  # Update with your Arduino port
    servo_x = board.get_pin('d:9:s')  # Servo connected to pin 9
    servo_y = board.get_pin('d:10:s')  # Servo connected to pin 10
except Exception as e:
    print(f"Error connecting to Arduino: {str(e)}")
    print("Please check the connection and COM port.")
    sys.exit(1)

showLandMarks = True
soundDetected = True

cap_width = 800
cap_height = 600
min_detection_confidence = 0.8
min_tracking_confidence = 0.5

esp_camera_url = 'http://192.168.254.137/800x600.mjpeg'

# Initialize previous positions and smooth factor
prev_center_x, prev_center_y = cap_width // 2, cap_height // 2
prev_servoX, prev_servoY = 90, 90  # Start with servos centered
smooth_factor = 0.1

frame_timeout = 5  # Timeout for frame in seconds

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
    number_per_class = 100
    num_to_log = -1  # Initialize to -1

    global prev_center_x, prev_center_y, prev_servoX, prev_servoY

    frame_queue = Queue(maxsize=1)
    thread = Thread(target=frame_reader, args=(frame_queue,))
    thread.daemon = True
    thread.start()

    last_frame_time = time.time()
    while True:
        try:
            fps = cvFpsCalc.get()

            key = cv.waitKey(10)
            if key == 27:  # ESC
                break
            number, mode, num_to_log = select_mode(key, mode, num_to_log)

            if frame_queue.empty():
                if time.time() - last_frame_time > frame_timeout:
                    print("Frame timeout detected. Restarting...")
                    cv.destroyAllWindows()
                    cap.release()
                    time.sleep(4)
                    return  # Exit the function to allow for a clean restart
                continue

            image = frame_queue.get()
            last_frame_time = time.time()

            if image is not None:
                image = cv.resize(image, (cap_width, cap_height))
                image = cv.flip(image, 1)
                debug_image = image.copy()

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
                            logging_thread = threading.Thread(target=logging_function, args=(
                                num_to_log, mode, pre_processed_landmark_list, pre_processed_point_history_list,
                                number_per_class))
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
                        debug_image = draw_info_text(debug_image, brect, handedness,
                                                     keypoint_classifier_labels[hand_sign_id],
                                                     point_history_classifier_labels[most_common_fg_id[0][0]])

                        # Move servo based on hand position
                        center_x = (brect[0] + brect[2]) // 2
                        center_y = (brect[1] + brect[3]) // 2
                        move_servo(center_x, center_y)
                        draw_crosshair(debug_image, (center_x, center_y))
                else:
                    point_history.append([0, 0])

                debug_image = draw_point_history(debug_image, point_history)
                debug_image = draw_info(debug_image, fps, mode, number)

                cv.imshow('Hand Gesture Recognition', debug_image)
        except Exception as e:
            print(f"Error in main loop: {str(e)}")
            print(f"Error occurred on line {sys.exc_info()[-1].tb_lineno}")
            print("Restarting in 5 seconds...")
            cv.destroyAllWindows()
            cap.release()
            time.sleep(5)
            return  # Exit the function to allow for a clean restart

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


def frame_reader(frame_queue):
    max_retries = 5
    retry_count = 0
    while retry_count < max_retries:
        try:
            stream = requests.get(esp_camera_url, stream=True, timeout=10)
            if stream.status_code != 200:
                raise Exception(f"Unable to open video stream. Status code: {stream.status_code}")

            bytes_data = b''
            for chunk in stream.iter_content(chunk_size=1024):
                bytes_data += chunk
                a = bytes_data.find(b'\xff\xd8')  # JPEG start
                b = bytes_data.find(b'\xff\xd9')  # JPEG end

                if a != -1 and b != -1:
                    jpg = bytes_data[a:b + 2]
                    bytes_data = bytes_data[b + 2:]
                    img = cv.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv.IMREAD_COLOR)
                    if frame_queue.full():
                        frame_queue.get_nowait()
                    frame_queue.put(img)

        except Exception as e:
            print(f"Error in frame_reader: {str(e)}")
            retry_count += 1
            print(f"Retrying... ({retry_count}/{max_retries})")
            time.sleep(5)

    print("Max retries reached. Exiting frame_reader.")


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


if __name__ == '__main__':
    while True:
        main()
        print("Restarting main function...")