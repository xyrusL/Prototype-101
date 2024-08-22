import csv
import mediapipe as mp
import pygame
import time
import threading
import speech_recognition as sr
import pyttsx3
import tkinter as tk
from utils.cvfpscalc import CvFpsCalc
from subPart.keyPointClassifier import KeyPointClassifier
from subPart.draw_landmarks import draw_landmarks
from subPart.bounding_rect import calc_bounding_rect
from subPart.process import *
from utils.menu import MenuInterface, return_global_values
from utils.config import (
    showLandMarks, soundDetected, letter_confidence_time, voice_version, 
    show_face_detection, show_body_detection, camera_id, 
    cap_width, cap_height, min_detection_confidence, min_tracking_confidence
)

# Do not change the values below
recognized_speech_text = ""
speech_recognition_active = False
stop_speech_recognition = False
speech_thread = None
automatic_listening = False
menu_is_open = False

# New Function for Face Detection
def process_face_detection(image, face_mesh):
    results = face_mesh.process(image)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw face landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1))
    return image

# New Function for Body Detection
def process_body_detection(image, pose):
    results = pose.process(image)
    if results.pose_landmarks:
        # Draw pose landmarks
        mp.solutions.drawing_utils.draw_landmarks(
            image=image,
            landmark_list=results.pose_landmarks,
            connections=mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
    return image

# Setup speech recognition
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

# Setup speech synthesis
def start_speech_recognition():
    global speech_thread, stop_speech_recognition
    stop_speech_recognition = False
    if speech_thread is None or not speech_thread.is_alive():
        speech_thread = threading.Thread(target=recognize_speech)
        speech_thread.start()

# Stop speech recognition
def stop_speech_recognition_thread():
    global stop_speech_recognition
    stop_speech_recognition = True
    if speech_thread is not None:
        speech_thread.join()

# Speech synthesis
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

# Main
def main():
    global showLandMarks, soundDetected, letter_confidence_time, voice_version, help_show
    global show_face_detection, show_body_detection, camera_id, cap_width, cap_height
    global min_detection_confidence, min_tracking_confidence, speech_recognition_active

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
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=min_detection_confidence)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)

    right_hand_model_path = 'ML Models/rightHandModel.tflite'
    left_hand_model_path = 'ML Models/leftHandModel.tflite'

    right_hand_classifier = KeyPointClassifier(model_path=right_hand_model_path)
    left_hand_classifier = KeyPointClassifier(model_path=left_hand_model_path)

    with open('ML Models/rightHand.csv', encoding='utf-8-sig') as f:
        right_hand_labels = [row[0] for row in csv.reader(f)]

    with open('ML Models/leftHand.csv', encoding='utf-8-sig') as f:
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

    while True:
        fps = cvFpsCalc.get()
        key = cv.waitKey(10)
        if key == 27 or key == ord('q'):  # esc or q to exit
            cap.release()
            cv.destroyAllWindows()
            break
        elif key == ord('d'):
            word_sentence_active = not word_sentence_active
            if word_sentence_active:
                speech_recognition_active = False
                automatic_listening = False
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
            word_sentence_active = False
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
                print("Listening")
            else:
                recognized_speech_text = ""
                print("Recognizing")
        elif key == ord('r'):
            if word_sentence_active and sentence:
                speak_text(sentence)
            elif speech_recognition_active and recognized_speech_text:
                speak_text(recognized_speech_text)
        elif key == ord('m'):
            if not menu_is_open:
                root = tk.Tk()
                menu = MenuInterface(root)
                root.mainloop()

                value = return_global_values()
                showLandMarks = value.showLandMarks
                soundDetected = value.soundDetected
                letter_confidence_time = value.letter_confidence_time
                voice_version = value.voice_version
                show_body_detection = value.show_body_detection
                show_face_detection = value.show_face_detection
                camera_id = value.camera_id
                cap_width = value.cap_width
                cap_height = value.cap_height
                min_detection_confidence = value.min_detection_confidence
                min_tracking_confidence = value.min_tracking_confidence
                background_color = value.background_color
                hand_detected_color_bg = value.hand_detected_color_bg

                # Update camera settings with new values
                cap.release()  
                cap = cv.VideoCapture(camera_id)  
                cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
                cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

                changeBg(background_color, "textbg")
                changeBg(hand_detected_color_bg, "handetectedbg")

        ret, image = cap.read()
        if not ret:
            break

        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        # Process Face and Body Detection
        if show_face_detection:
             debug_image = process_face_detection(debug_image, face_mesh)
        if show_body_detection:
            debug_image = process_body_detection(debug_image, pose)

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                # Check if hand is right or left
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

                    if letter_start_time is None or current_letter != previous_letter:
                        letter_start_time = time.time()
                        previous_letter = current_letter
                    elif time.time() - letter_start_time >= consistency_threshold:
                        word = word[:cursor_position] + current_letter + word[cursor_position:]
                        cursor_position += 1
                        letter_start_time = None

        debug_image = draw_info(debug_image, fps)
        debug_image = draw_info_detected_text(debug_image, current_label, formatted_confidence)

        if word_sentence_active:
            debug_image = draw_word_sentence(debug_image, word, sentence, cursor_position)

        if speech_recognition_active:
            debug_image = draw_recognized_speech(debug_image, recognized_speech_text)

        cv.imshow('ALPHANUMERIC RECOGNITION BY CS SAN MATEO 2024 - Dev. Paul', debug_image)

    cap.release()
    cv.destroyAllWindows()

# Play sound
def play_sound_for_hand_sign(hand_sign_label):
    def play_sound():
        file_path = f'Voices/version1/{hand_sign_label}.mp3'

        if voice_version == 2:
            file_path = f'Voices/version2/{hand_sign_label}.mp3'

        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

    play_sound()


if __name__ == '__main__':
    main()