import cv2 as cv
import copy
import itertools

def draw_info_text(image, brect, handedness, hand_sign_text, confidence):
    if brect is not None:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 40), (139, 0, 0), -1)
        info_text = handedness.classification[0].label
        if hand_sign_text:
            info_text += ':' + hand_sign_text + ' ' + str(confidence)
        font_scale = 1.0
        thickness = 2
        cv.putText(image, info_text, (brect[0] + 5, brect[1] - 8), cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv.LINE_AA)
    return image

def draw_info_detected_text(image, hand_sign_label, confidence_percentage):
    # Prepare the text for display
    detected_text = "Machine Learning Detected: " + hand_sign_label
    confidence_text = f"Predictive Confidence: {confidence_percentage}"

    # Calculate text sizes
    detected_text_size = cv.getTextSize(detected_text, cv.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    confidence_text_size = cv.getTextSize(confidence_text, cv.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

    # Calculate text positions
    text_x, text_y = 10, 60
    confidence_text_y = text_y + detected_text_size[1] + 10

    # Draw rectangles for the text backgrounds
    cv.rectangle(image, (text_x - 5, text_y - detected_text_size[1] - 5), (text_x + detected_text_size[0] + 5, text_y + 5), (50, 50, 50), -1)
    cv.rectangle(image, (text_x - 5, confidence_text_y - confidence_text_size[1] - 5), (text_x + confidence_text_size[0] + 5, confidence_text_y + 5), (50, 50, 50), -1)

    # Draw the text on the image
    cv.putText(image, detected_text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, detected_text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

    cv.putText(image, confidence_text, (text_x, confidence_text_y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, confidence_text, (text_x, confidence_text_y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)

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
    text = "FPS:" + str(fps)
    text_size = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x, text_y = 10, 30
    cv.rectangle(image, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), (50, 50, 50), -1)
    cv.putText(image, text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
    return image

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (139, 0, 0), 2)
    return image


def draw_help_info(image):
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    color = (255, 255, 255)
    margin = 20  # Margin from the right edge
    y_margin = 20  # Line spacing

    text = [
        "q / esc - Quit",
        "h - Help (hide/show)",
        "d - Toggle word/sentence mode",
        "f - Delete character",
        "r - Read sentence/speech",
        "Space (Hand Sign) - Put the word in the sentence",
        "v - Voice version",
    ]

    x_pos = image.shape[1] - margin
    y_pos = image.shape[0] // 2 - (len(text) * y_margin) // 2

    for i, line in enumerate(text):
        y = y_pos + i * y_margin
        text_size = cv.getTextSize(line, font, font_scale, font_thickness)[0]
        x = x_pos - text_size[0]
        cv.putText(image, line, (x, y), font, font_scale, color, font_thickness, cv.LINE_AA)

    return image