import numpy as np
import cv2 as cv
import itertools
import copy
import argparse
# Argument parsing
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=1280)
    parser.add_argument("--height", help='cap height', type=int, default=720)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", help='min_detection_confidence', type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", help='min_tracking_confidence', type=int, default=0.5)

    args = parser.parse_args()
    return args
# Bounding box calculation
def bound_rect_calc(image, landmarks):
    # Get the width and height of the image
    image_width, image_height = image.shape[1], image.shape[0]

    # Create an empty numpy array to store the landmark coordinates
    landmark_array = np.empty((0, 2), int)

    # Loop through the landmarks and add their coordinates to the landmark_array
    for _, landmark in enumerate(landmarks.landmark):
        # Get the x and y coordinates of the landmark, ensuring they are within the image dimensions
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        # Convert the landmark coordinates to a numpy array
        landmark_point = [np.array((landmark_x, landmark_y))]

        # Append the landmark coordinates to the landmark_array
        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    # Calculate the bounding rectangle around the landmark coordinates
    x, y, w, h = cv.boundingRect(landmark_array)

    # Return the coordinates of the bounding rectangle
    return [x, y, x + w, y + h]

# Calculate the landmark list
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

# Function to preprocess the landmark
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
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

# Function to preprocess the point history
def pre_process_point_history(image, point_history):
    # Get the width and height of the image
    image_width, image_height = image.shape[1], image.shape[0]

    # Create a copy of the point history
    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        # Update the x and y coordinates relative to the base point
        temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(itertools.chain.from_iterable(temp_point_history))

    return temp_point_history