import csv
import mediapipe as mp
import pygame
import time
import threading
import speech_recognition as sr
import pyttsx3
from utils import CvFpsCalc
from subPart.keyPointClassifier import KeyPointClassifier
from subPart.args import get_args
from subPart.draw_landmarks import draw_landmarks
from subPart.bounding_rect import calc_bounding_rect
from subPart.process import *

showLandMarks = True
soundDetected = True
letter_confidence_time = 1.5

cap_width = 1280
cap_height = 720
min_detection_confidence = 0.7
min_tracking_confidence = 0.5

recognized_speech_text = ""
speech_recognition_active = False
stop_speech_recognition = False
speech_thread = None

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

def start_speech_recognition():
    global speech_thread
    if speech_thread is None or not speech_thread.is_alive():
        speech_thread = threading.Thread(target=recognize_speech)
        speech_thread.start()

def speak_text(text):
    engine = pyttsx3.init()

    def _speak():
        engine.say(text)
        engine.runAndWait()

    thread = threading.Thread(target=_speak)
    thread.start()

def main():
    global speech_recognition_active, stop_speech_recognition, recognized_speech_text, speech_thread
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
        right_hand_labels = [row[0] for row in csv.reader(f)]

    with open('Model/leftHand.csv', encoding='utf-8-sig') as f:
        left_hand_labels = [row[0] for row in csv.reader(f)]

    cvFpsCalc = CvFpsCalc(buffer_len=10)

    previous_label = ""
    current_label = ""

    sentence = ""
    word = ""
    previous_letter = ""
    cursor_position = 0

    letter_start_time = None
    consistency_threshold = letter_confidence_time

    word_sentence_active = False

    while True:
        fps = cvFpsCalc.get()
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        elif key == ord('d'):
            word_sentence_active = not word_sentence_active
            if word_sentence_active:
                speech_recognition_active = False
                stop_speech_recognition = True
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
            if speech_recognition_active:
                word_sentence_active = False
                stop_speech_recognition = False
                start_speech_recognition()
            else:
                stop_speech_recognition = True
                recognized_speech_text = ""
        elif key == ord('r'):
            if word_sentence_active and sentence:
                speak_text(sentence)
            elif speech_recognition_active and recognized_speech_text:  # Speak recognized speech if in STT mode
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

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                if handedness.classification[0].label == 'Right':
                    hand_sign_id = right_hand_classifier(pre_processed_landmark_list)
                    current_label = right_hand_labels[hand_sign_id]
                else:
                    hand_sign_id = left_hand_classifier(pre_processed_landmark_list)
                    current_label = left_hand_labels[hand_sign_id]

                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_info_text(debug_image, brect, handedness, current_label, current_label)

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

                    if letter_start_time is None or current_letter != previous_letter:
                        letter_start_time = time.time()
                        previous_letter = current_letter
                    elif time.time() - letter_start_time >= consistency_threshold:
                        word = word[:cursor_position] + current_letter + word[cursor_position:]
                        cursor_position += 1
                        letter_start_time = None

        debug_image = draw_info(debug_image, fps)
        debug_image = draw_info_detected_text(debug_image, current_label)

        if word_sentence_active:
            debug_image = draw_word_sentence(debug_image, word, sentence, cursor_position)

        if speech_recognition_active:
            debug_image = draw_recognized_speech(debug_image, recognized_speech_text)

        cv.imshow('FSL RECOGNITION BY CS SAN MATEO - DEV.PAUL', debug_image)

    cap.release()
    cv.destroyAllWindows()

def play_sound_for_hand_sign(hand_sign_label):
    def play_sound():
        file_path = f'voices/{hand_sign_label}.mp3'
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

    play_sound()

if __name__ == '__main__':
    main()