import cv2 as cv

def draw_landmarks(image, landmark_point):
    def draw_line(image, point1, point2):
        cv.line(image, tuple(point1), tuple(point2), (0, 0, 0), 6)
        cv.line(image, tuple(point1), tuple(point2), (0, 0, 255), 2)

    def draw_circle(image, point, radius=5):
        cv.circle(image, tuple(point), radius, (0, 255, 0), -1)
        cv.circle(image, tuple(point), radius, (0, 0, 0), 1)

    finger_indices = [
        (2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12),
        (13, 14, 15, 16), (17, 18, 19, 20)
    ]

    if len(landmark_point) > 0:
        # Draw fingers
        for finger in finger_indices:
            for i in range(len(finger) - 1):
                draw_line(image, landmark_point[finger[i]], landmark_point[finger[i+1]])

        # Draw palm
        palm_indices = [(0, 1), (1, 2), (2, 5), (5, 9), (9, 13), (13, 17), (17, 0)]
        for idx in palm_indices:
            draw_line(image, landmark_point[idx[0]], landmark_point[idx[1]])

    # Draw key points
    key_point_radii = {4: 8, 8: 8, 12: 8, 16: 8, 20: 8}
    for i, point in enumerate(landmark_point):
        draw_circle(image, point, radius=key_point_radii.get(i, 5))

    return image
