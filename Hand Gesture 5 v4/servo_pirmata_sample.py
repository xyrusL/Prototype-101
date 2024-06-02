import cv2
import mediapipe as mp
import numpy as np
import pyfirmata
import urllib.request
import time

# Initialize mediapipe and pyfirmata
mpHand = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHand.Hands(min_detection_confidence=0.8)
board = pyfirmata.Arduino("COM3")
servo_pin = board.get_pin('d:9:s')  # Horizontal movement

# Initialize camera with IP stream using urllib
stream_url = 'http://192.168.254.137/1280x720.mjpeg'
stream = urllib.request.urlopen(stream_url)
bytes = b''

# Function to move the servo smoothly with variable speed
def move_servo_smoothly(pin, start_value, end_value, damping_factor=5):
    step = 1 if end_value > start_value else -1
    distance = abs(end_value - start_value)
    max_speed = 0.005
    base_speed = 0.05
    speed = max(base_speed - (distance * 0.001), max_speed)

    damped_end_value = start_value + (end_value - start_value) / damping_factor

    for value in range(start_value, int(damped_end_value), step):
        pin.write(value)
        time.sleep(speed)
        time.sleep(0.01)  # Add an additional delay after each step

# Initial servo position
current_servo_value = 90  # Start from the middle position

while True:
    bytes += stream.read(1024)
    a = bytes.find(b'\xff\xd8')
    b = bytes.find(b'\xff\xd9')
    if a != -1 and b != -1:
        jpg = bytes[a:b + 2]
        bytes = bytes[b + 2:]
        img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.flip(img, 1)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)

            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    mpDraw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONS)
                    # Track wrist (id=0) or palm center (id=9)
                    lm = handLms.landmark[0]  # Using wrist landmark for example
                    h, w, c = img.shape
                    cx = int(lm.x * w)

                    # Map cx to servo range
                    new_servo_value = int(np.interp(cx, [0, w], [0, 180]))

                    # Move the servo smoothly
                    if abs(new_servo_value - current_servo_value) > 1:  # Only move if there's a significant change
                        move_servo_smoothly(servo_pin, current_servo_value, new_servo_value)
                        current_servo_value = new_servo_value  # Update the current position

                    # Optionally draw the tracking point
                    cv2.circle(img, (cx, int(lm.y * h)), 15, (0, 255, 0), cv2.FILLED)

            # Draw crosshair in the center of the image
            center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
            cv2.line(img, (center_x - 20, center_y), (center_x + 20, center_y), (0, 0, 255), 2)
            cv2.line(img, (center_x, center_y - 20), (center_x, center_y + 20), (0, 0, 255), 2)

            # Display the image
            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.01)  # Reduce the delay to improve smoothness
        else:
            print("Failed to decode the image from the stream.")
    else:
        print("Received an empty frame from the stream.")

cv2.destroyAllWindows()
board.exit()
