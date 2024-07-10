import csv
import mediapipe as mp
import pygame
import time
import threading
import speech_recognition as sr
import pyttsx3
import pyfirmata
from utils import CvFpsCalc
from subPart.keyPointClassifier import KeyPointClassifier
from subPart.draw_landmarks import draw_landmarks
from subPart.bounding_rect import calc_bounding_rect
from subPart.process import *

# Global configuration
showLandMarks = True  # Change to True if you want to see the landmarks
soundDetected = True  # Change to True if you want to hear the sound
letter_confidence_time = 1.5  # Time in seconds for the letter to be recognized
voice_version = 1  # 1 for voice 1, 2 for voice 2

camera_id = 1  # Change to 0 if you want to use your webcam
cap_width = 1280  # Change the width of the frame
cap_height = 720  # Change the height of the frame
min_detection_confidence = 0.7  # Change the detection confidence value
min_tracking_confidence = 0.5  # Change the tracking confidence value

# Do not change the values below
recognized_speech_text = ""
speech_recognition_active = False
stop_speech_recognition = False
speech_thread = None
automatic_listening = False

# Initialize pyFirmata board
board = pyfirmata.Arduino('COM5')  # Change this to the appropriate port
servo_pin = 9  # Change this to the appropriate pin
servo = board.get_pin(f'd:{servo_pin}:s')

# Initialize servo to the center position (90 degrees)
servo.write(90)

# Function to map the range of values
def map_range(value, in_min, in_max, out_min, out_max):
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

# Function to move the servo smoothly to the center
def move_servo_to_center(wrist_coordinates, frame_width, frame_height, previous_x, smooth_factor=0.1):
    # Calculate the center of the frame
    center_x, center_y = frame_width // 2, frame_height // 2
    wrist_x, wrist_y = wrist_coordinates

    # Smooth movement by using the previous x value
    wrist_x = previous_x * (1 - smooth_factor) + wrist_x * smooth_factor

    # Map the wrist x-coordinate to the servo angle (0-180 degrees)
    servo_angle = map_range(wrist_x, 0, frame_width, 0, 200)
    servo.write(servo_angle)

    return wrist_x

# Speech recognition function
def recognize_speech():
    global recognized_speech_text, stop_speech_recognition

    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        while not stop_speech_recognition:
            audio_data = r.record(source, duration=5)
            print("Recognizing...")
            try:
                text = r.recognize_google(audio_data)
                print("You said: " + text)
                recognized_speech_text = text
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
                recognized_speech_text = ""
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
                recognized_speech_text = ""
            if stop_speech_recognition:
                break

# Start speech recognition in a separate thread
def start_speech_recognition():
    global speech_thread
    if speech_thread is None or not speech_thread.is_alive():
        speech_thread = threading.Thread(target=recognize_speech)
        speech_thread.start()

# Stop speech recognition thread
def stop_speech_recognition_thread():
    global stop_speech_recognition
    stop_speech_recognition = True
    if speech_thread is not None:
        speech_thread.join()

# Text-to-speech function
def speak_text(text):
    engine = pyttsx3.init()

    def _speak():
        engine.say(text)
        engine.runAndWait()

    thread = threading.Thread(target=_speak)
    thread.start()

# Function to calculate confidence accuracy
def calculate_confidence(landmark_list, hand_classifier):
    confidences = hand_classifier.get_confidences(landmark_list)
    max_confidence = max(confidences)
    return max_confidence

# Main function
def main():
    global speech_recognition_active, stop_speech_recognition, recognized_speech_text, speech_thread, automatic_listening, confidence, handedness, brect
    cap_device = camera_id
    use_static_image_mode = 'store_true'
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
        right_hand_labels = [row[0] for row in csv.reader(f)]

    with open('Model/leftHand.csv', encoding='utf-8-sig') as f:
        left_hand_labels = [row[0] for row in csv.reader(f)]

    cvFpsCalc = CvFpsCalc(buffer_len=10)

    previous_label = ""
    current_label = ""
    formatted_confidence = ""

    sentence = ""
    word = ""
    previous_letter = ""
    cursor_position = 0

    letter_start_time = None
    consistency_threshold = letter_confidence_time

    word_sentence_active = False

    previous_x = cap_width // 2  # Initialize previous_x to the center of the frame

    while True:
        fps = cvFpsCalc.get()
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        elif key == ord('d'):
            word_sentence_active = not word_sentence_active
            if word_sentence_active:
                speech_recognition_active = False
                stop_speech_recognition_thread()
                recognized_speech_text = ""
                word = ""
                sentence = ""
                cursor_position = 0
        elif key == ord('f'):
            if cursor_position > 0:
                word = word[:cursor_position - 1] + word[cursor_position:]
                cursor_position -= 1
        elif key == ord('s'):
            speech_recognition_active = not speech_recognition_active
            automatic_listening = False
            stop_speech_recognition_thread()
            recognized_speech_text = ""
        elif key == ord('a') and speech_recognition_active:
            automatic_listening = not automatic_listening
            if automatic_listening:
                stop_speech_recognition = False
                start_speech_recognition()
                print("Automatic listening ON")
            else:
                stop_speech_recognition_thread()
                recognized_speech_text = ""
                print("Automatic listening OFF")
        elif key == ord('l') and speech_recognition_active and not automatic_listening:
            stop_speech_recognition = not stop_speech_recognition
            if not stop_speech_recognition:
                start_speech_recognition()
                print("Manual listening ON")
            else:
                recognized_speech_text = ""
                print("Manual listening OFF")
        elif key == ord('r'):
            if word_sentence_active and sentence:
                speak_text(sentence)
            elif speech_recognition_active and recognized_speech_text:
                speak_text(recognized_speech_text)

        ret, image = cap.read()
        if not ret:
            break

        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        wrist_coordinates = None
        hand_in_frame = False
        brect = None
        handedness = None  # Initialize handedness

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                # Check if the hand is right or left
                if handedness.classification[0].label == 'Right':
                    hand_sign_id = right_hand_classifier(pre_processed_landmark_list)
                    current_label = right_hand_labels[hand_sign_id]
                    confidence = calculate_confidence(pre_processed_landmark_list, right_hand_classifier) * 100  # Calculate confidence
                else:
                    hand_sign_id = left_hand_classifier(pre_processed_landmark_list)
                    current_label = left_hand_labels[hand_sign_id]
                    confidence = calculate_confidence(pre_processed_landmark_list, left_hand_classifier) * 100  # Calculate confidence

                formatted_confidence = f"{confidence:.1f}%"
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_info_text(debug_image, brect, handedness, current_label, formatted_confidence)

                if showLandMarks:
                    debug_image = draw_landmarks(debug_image, landmark_list)

                if soundDetected:
                    if current_label != previous_label:
                        play_sound_for_hand_sign(current_label)
                        previous_label = current_label

                if handedness.classification[0].label == 'Right' and word_sentence_active:
                    hand_sign_id = right_hand_classifier(pre_processed_landmark_list)
                    current_letter = right_hand_labels[hand_sign_id]

                    if current_letter == 'Space':
                        if word:
                            sentence += word + ' '
                            word = ""
                            cursor_position = 0
                        continue

                    # Consistency check
                    if letter_start_time is None or current_letter != previous_letter:
                        letter_start_time = time.time()
                        previous_letter = current_letter
                    elif time.time() - letter_start_time >= consistency_threshold:
                        word = word[:cursor_position] + current_letter + word[cursor_position:]
                        cursor_position += 1
                        letter_start_time = None

                # Wrist coordinates
                wrist_coordinates = (landmark_list[0][0], landmark_list[0][1])
                hand_in_frame = True

        #  Check if wrist coordinates are valid and hand is in frame
        if wrist_coordinates is not None and hand_in_frame:
            previous_x = move_servo_to_center(wrist_coordinates, cap_width, cap_height, previous_x)

        debug_image = draw_info(debug_image, fps)
        debug_image = draw_info_detected_text(debug_image, current_label, formatted_confidence)

        if word_sentence_active:
            debug_image = draw_word_sentence(debug_image, word, sentence, cursor_position)

        if speech_recognition_active:
            debug_image = draw_recognized_speech(debug_image, recognized_speech_text)

        cv.imshow('FSL RECOGNITION BY CS SAN MATEO - DEV.PAUL', debug_image)

    cap.release()
    cv.destroyAllWindows()


# Play sound for hand sign
def play_sound_for_hand_sign(hand_sign_label):
    def play_sound():
        file_path = f'voices/{hand_sign_label}.mp3'
        if voice_version == 2:
            file_path = f'voices/{hand_sign_label}.mp3'

        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

    play_sound()

if __name__ == '__main__':
    main()
