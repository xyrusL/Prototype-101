import cv2 as cv

# Define colors as global variables
HAND_COLOR = (0, 0, 0)
HAND_CIRCLE_COLOR = (148, 0, 211)
HAND_THICKNESS = 6
KEY_POINT_RADIUS = 5
KEY_POINT_CIRCLE_THICKNESS = 1
FINGER_TIP_RADIUS = 8

def draw_landmarks(image, landmark_point):
    global HAND_COLOR, HAND_THICKNESS, HAND_CIRCLE_COLOR, KEY_POINT_RADIUS, KEY_POINT_CIRCLE_THICKNESS, FINGER_TIP_RADIUS

    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), HAND_COLOR, HAND_THICKNESS)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), HAND_CIRCLE_COLOR, 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), HAND_COLOR, HAND_THICKNESS)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), HAND_CIRCLE_COLOR, 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), HAND_COLOR, HAND_THICKNESS)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), HAND_CIRCLE_COLOR, 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), HAND_COLOR, HAND_THICKNESS)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), HAND_CIRCLE_COLOR, 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), HAND_COLOR, HAND_THICKNESS)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), HAND_CIRCLE_COLOR, 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), HAND_COLOR, HAND_THICKNESS)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), HAND_CIRCLE_COLOR, 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), HAND_COLOR, HAND_THICKNESS)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), HAND_CIRCLE_COLOR, 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), HAND_COLOR, HAND_THICKNESS)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), HAND_CIRCLE_COLOR, 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), HAND_COLOR, HAND_THICKNESS)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), HAND_CIRCLE_COLOR, 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), HAND_COLOR, HAND_THICKNESS)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), HAND_CIRCLE_COLOR, 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), HAND_COLOR, HAND_THICKNESS)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), HAND_CIRCLE_COLOR, 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), HAND_COLOR, HAND_THICKNESS)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), HAND_CIRCLE_COLOR, 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), HAND_COLOR, HAND_THICKNESS)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), HAND_CIRCLE_COLOR, 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), HAND_COLOR, HAND_THICKNESS)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), HAND_CIRCLE_COLOR, 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), HAND_COLOR, HAND_THICKNESS)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), HAND_CIRCLE_COLOR, 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), HAND_COLOR, HAND_THICKNESS)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), HAND_CIRCLE_COLOR, 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), HAND_COLOR, HAND_THICKNESS)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), HAND_CIRCLE_COLOR, 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), HAND_COLOR, HAND_THICKNESS)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), HAND_CIRCLE_COLOR, 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), HAND_COLOR, HAND_THICKNESS)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), HAND_CIRCLE_COLOR, 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), HAND_COLOR, HAND_THICKNESS)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), HAND_CIRCLE_COLOR, 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), HAND_COLOR, HAND_THICKNESS)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), HAND_CIRCLE_COLOR, 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index in [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]:
            cv.circle(image, (landmark[0], landmark[1]), KEY_POINT_RADIUS, HAND_CIRCLE_COLOR, -1)
            cv.circle(image, (landmark[0], landmark[1]), KEY_POINT_RADIUS, HAND_COLOR, KEY_POINT_CIRCLE_THICKNESS)
        elif index in [4, 8, 12, 16, 20]:
            cv.circle(image, (landmark[0], landmark[1]), FINGER_TIP_RADIUS, HAND_CIRCLE_COLOR, -1)
            cv.circle(image, (landmark[0], landmark[1]), FINGER_TIP_RADIUS, HAND_COLOR, KEY_POINT_CIRCLE_THICKNESS)

    return image