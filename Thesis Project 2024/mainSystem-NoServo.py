import csv
import mediapipe as mp
import pygame
import numpy as np
import requests
import time
from threading import Thread
from queue import Queue
from utils import CvFpsCalc
from subPart.keyPointClassifier import KeyPointClassifier
from subPart.args import get_args
from subPart.draw_landmarks import draw_landmarks
from subPart.bounding_rect import calc_bounding_rect
from subPart.process import *

showLandMarks = True
soundDetected = True

cap_width = 800
cap_height = 600
min_detection_confidence = 0.8
min_tracking_confidence = 0.5

esp_camera_url = 'http://192.168.254.137/1280x1024.mjpeg'
frame_timeout = 5  # Timeout for frame in seconds

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

            if frame_queue.empty():
                if time.time() - last_frame_time > frame_timeout:
                    play_sound_for_hand_sign("timeout-restart")
                    print("Frame timeout detected. Restarting...")
                    cv.destroyAllWindows()
                    time.sleep(4)
                    main()
                continue

            img = frame_queue.get()
            last_frame_time = time.time()  # Update last frame time

            if img is not None:
                img = cv.resize(img, (cap_width, cap_height))

                image = cv.flip(img, 1)
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

                # Draw the crosshair
                draw_crosshair(debug_image, (cap_width // 2, cap_height // 2))

                cv.imshow('FSL RECOGNITION BY CS SAN MATEO', debug_image)

        except Exception as e:
            print(f"Error: {str(e)}")
            print(f"\033[92mTrying to reconnect...\033[0m")
            cv.waitKey(1000)
            continue

    cv.destroyAllWindows()

def frame_reader(frame_queue):
    stream = requests.get(esp_camera_url, stream=True)
    if stream.status_code != 200:
        print(f"Error: Unable to open video stream from {esp_camera_url}")
        return

    bytes_data = b''
    for chunk in stream.iter_content(chunk_size=1024):
        bytes_data += chunk
        a = bytes_data.find(b'\xff\xd8')  # JPEG start
        b = bytes_data.find(b'\xff\xd9')  # JPEG end

        if a != -1 and b != -1:
            jpg = bytes_data[a:b+2]
            bytes_data = bytes_data[b+2:]
            img = cv.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv.IMREAD_COLOR)
            if frame_queue.full():
                frame_queue.get_nowait()
            frame_queue.put(img)

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
