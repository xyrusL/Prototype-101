import numpy as np
import cv2 as cv

# Function to calculate the bounding rectangle for a set of hand landmarks on an image
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)

    # Iterate over each landmark and convert it to pixel coordinates
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    # Calculate the bounding rectangle based on the landmark points
    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]
