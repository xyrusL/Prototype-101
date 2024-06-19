import itertools
import copy
import cv2 as cv

# Function to calculate landmark list from image and landmarks
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

# Function to preprocess landmark list
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


# Function to preprocess point history
def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history

# Function to draw information text on the image
def draw_info_text(image, brect, handedness, hand_sign_text, hand_detected, showDetected):
    # Set font scale and thickness
    font_scale = 0.9
    font_thickness = 2

    # Calculate the text size
    info_text = handedness.classification[0].label
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text

    text_size, _ = cv.getTextSize(info_text, cv.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

    # Draw the background rectangle
    cv.rectangle(image, (brect[0], brect[1] - text_size[1] - 10), (brect[0] + text_size[0] + 10, brect[1]), (0, 0, 0), -1)

    # Put the text on the image
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 5), cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv.LINE_AA)

    # Display the hand detected information
    if showDetected:
        if hand_detected == "Right":
            cv.putText(image, "Right Hand: " + hand_sign_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
                       font_thickness, cv.LINE_AA)
            cv.putText(image, "Right Hand: " + hand_sign_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, font_scale,
                       (255, 255, 255), font_thickness - 1, cv.LINE_AA)
        elif hand_detected == "Left":
            cv.putText(image, "Left Hand: " + hand_sign_text, (10, 90), cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
                       font_thickness, cv.LINE_AA)
            cv.putText(image, "Left Hand: " + hand_sign_text, (10, 90), cv.FONT_HERSHEY_SIMPLEX, font_scale,
                       (255, 255, 255), font_thickness - 1, cv.LINE_AA)

    return image


# Function to draw point history on the image
def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image

# Function to draw information on the image
def draw_info(image, fps):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)
    return image


# Function to draw bounding rectangle on the image
def draw_bounding_rect(use_brect, image, brect, expand_ratio=1.2):
    if use_brect:
        # Calculate the center of the bounding box
        center_x = (brect[0] + brect[2]) // 2
        center_y = (brect[1] + brect[3]) // 2

        # Calculate the new width and height by expanding the original size
        width = (brect[2] - brect[0]) * expand_ratio
        height = (brect[3] - brect[1]) * expand_ratio

        # Calculate the new bounding box coordinates
        new_brect = [
            int(center_x - width // 2),
            int(center_y - height // 2),
            int(center_x + width // 2),
            int(center_y + height // 2)
        ]

        # Draw the expanded bounding box
        cv.rectangle(image, (new_brect[0], new_brect[1]), (new_brect[2], new_brect[3]), (0, 0, 0), 2)

    return image
