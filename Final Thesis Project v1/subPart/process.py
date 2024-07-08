import cv2 as cv
import copy
import itertools


def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
    info_text = handedness.classification[0].label
    if hand_sign_text:
        info_text += ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
               cv.LINE_AA)
    return image

def draw_info_detected_text(image, hand_sign_label):
    text = "Machine Learning Detected: " + hand_sign_label
    cv.putText(image, text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv.LINE_AA)
    return image

def draw_word_sentence(image, word, sentence, cursor_position):
    height, width, _ = image.shape
    word_with_cursor = word[:cursor_position] + '|' + word[cursor_position:]

    font_scale = 1
    font_thickness = 2

    word_text = f"WORD: {word_with_cursor}"
    sentence_text = f"SENTENCE: {sentence}"
    word_text_size = cv.getTextSize(word_text, cv.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
    sentence_text_size = cv.getTextSize(sentence_text, cv.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]

    word_rect_x1, word_rect_y1 = 10, height - 70
    word_rect_x2 = min(word_rect_x1 + word_text_size[0] + 20, width - 10)
    word_rect_y2 = word_rect_y1 - word_text_size[1] - 20

    sentence_rect_x1, sentence_rect_y1 = 10, height - 30
    sentence_rect_x2 = min(sentence_rect_x1 + sentence_text_size[0] + 20, width - 10)
    sentence_rect_y2 = sentence_rect_y1 - sentence_text_size[1] - 20

    cv.rectangle(image, (word_rect_x1, word_rect_y1), (word_rect_x2, word_rect_y2), (50, 50, 50), -1)
    cv.rectangle(image, (sentence_rect_x1, sentence_rect_y1), (sentence_rect_x2, sentence_rect_y2), (50, 50, 50), -1)

    cv.putText(image, word_text, (word_rect_x1 + 10, word_rect_y1 - 10), cv.FONT_HERSHEY_SIMPLEX,
               font_scale, (255, 255, 255), font_thickness, cv.LINE_AA)
    cv.putText(image, sentence_text, (sentence_rect_x1 + 10, sentence_rect_y1 - 10),
               cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv.LINE_AA)

    return image

def draw_recognized_speech(image, recognized_speech):
    height, width, _ = image.shape
    font_scale = 1
    font_thickness = 2

    speech_text = f"Recognized Speech: {recognized_speech}"
    speech_text_size = cv.getTextSize(speech_text, cv.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]

    speech_rect_x1, speech_rect_y1 = 10, height - 30
    speech_rect_x2 = min(speech_rect_x1 + speech_text_size[0] + 20, width - 10)
    speech_rect_y2 = speech_rect_y1 - speech_text_size[1] - 20

    cv.rectangle(image, (speech_rect_x1, speech_rect_y1), (speech_rect_x2, speech_rect_y2), (50, 50, 50), -1)
    cv.putText(image, speech_text, (speech_rect_x1 + 10, speech_rect_y1 - 10), cv.FONT_HERSHEY_SIMPLEX,
               font_scale, (255, 255, 255), font_thickness, cv.LINE_AA)

    return image

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = [
        [min(int(landmark.x * image_width), image_width - 1), min(int(landmark.y * image_height), image_height - 1)] for
        landmark in landmarks.landmark]
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0]

    for index, landmark_point in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(map(abs, temp_landmark_list))
    temp_landmark_list = [n / max_value for n in temp_landmark_list]

    return temp_landmark_list

def draw_info(image, fps):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
    return image

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image
