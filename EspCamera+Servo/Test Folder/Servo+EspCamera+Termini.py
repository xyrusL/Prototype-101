import cv2
import mediapipe as mp
import numpy as np
import pyfirmata
import urllib.request
from PIL import Image
from io import BytesIO

# Initialize MediaPipe Hands solution
mpHand = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHand.Hands(min_detection_confidence=0.8)

# ESP camera URL
url = "http://192.168.254.137/cam-hi.jpg"
ws, hs = 800, 600

# Initialize Arduino board and servo pins
port = "COM3"
board = pyfirmata.Arduino(port)
servo_pinX = board.get_pin('d:9:s')  # pin 9 Arduino
servo_pinY = board.get_pin('d:10:s')  # pin 10 Arduino

# Move servos to the middle position at the start
servo_pinX.write(90)
servo_pinY.write(90)

# Variables to store previous servo positions for smooth movement
prev_servoX = 90
prev_servoY = 90

# Center of the frame
center_x = ws // 2
center_y = hs // 2

# Initialize servoX and servoY
servoX = 90
servoY = 90

# Initialize target_status
target_status = "Target: No Hand Detected"

# Define a function to draw the terminator-style overlay
def draw_terminator_overlay(img, servoX, servoY, target_status, error_message=None):
    # Draw information boxes with values
    cv2.putText(img, f'Servo X: {servoX} deg', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, f'Servo Y: {servoY} deg', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Draw crosshair at the center
    cv2.line(img, (center_x - 20, center_y), (center_x + 20, center_y), (0, 0, 255), 2)
    cv2.line(img, (center_x, center_y - 20), (center_x, center_y + 20), (0, 0, 255), 2)

    # Display error message if any
    if error_message:
        cv2.rectangle(img, (40, hs - 100), (450, hs - 50), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, error_message, (50, hs - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(img, target_status, (50, hs - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

while True:
    try:
        # Fetch image from ESP camera
        response = urllib.request.urlopen(url, timeout=5)  # Set a timeout value (e.g., 5 seconds)
        img = Image.open(BytesIO(response.read()))
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (ws, hs))

        img = cv2.flip(img, 1)  # Flip the image horizontally
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        multiHandDetection = results.multi_hand_landmarks  # Hand Detection
        lmList = []

        if multiHandDetection:
            target_status = "Target: Hand Detected"
            # Hand Visualization
            for handLms in multiHandDetection:
                mpDraw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONS)

                # Get the bounding box for the hand
                bbox = cv2.boundingRect(np.array([[lm.x * ws, lm.y * hs] for lm in handLms.landmark]).astype(int))
                x, y, w, h = bbox
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red bounding box

                # Calculate center of the bounding box
                bbox_center_x = x + w // 2
                bbox_center_y = y + h // 2

                # Check if the bounding box is outside the frame
                if x < 0 or x + w > ws or y < 0 or y + h > hs:
                    # Trigger Algorithm: Move the servo to catch the hand
                    if x < 0:
                        servoX = 0
                    elif x + w > ws:
                        servoX = 180
                    else:
                        servoX = int(np.interp(bbox_center_x, [0, ws], [0, 180]))

                    if y < 0:
                        servoY = 180
                    elif y + h > hs:
                        servoY = 0
                    else:
                        servoY = int(np.interp(bbox_center_y, [0, hs], [0, 180]))

                else:
                    # Calculate the offset from the center of the bounding box to the center of the frame
                    offset_x = center_x - bbox_center_x
                    offset_y = center_y - bbox_center_y

                    # Convert position to degree value, adjusting for center offset
                    servoX = int(np.interp(center_x - offset_x, [0, ws], [0, 180]))
                    servoY = int(np.interp(center_y - offset_y, [0, hs], [0, 180]))

            # Smooth movement
            smooth_factor = 0.2
            servoX = int(prev_servoX + smooth_factor * (servoX - prev_servoX))
            servoY = int(prev_servoY + smooth_factor * (servoY - prev_servoY))
            prev_servoX, prev_servoY = servoX, servoY

            # Write to servo pins
            servo_pinX.write(servoX)
            servo_pinY.write(servoY)

            print(f'\033[94mHand Position x: {bbox_center_x} y: {bbox_center_y}\033[0m')  # Print in blue color
            print(f'\033[94mServo Value x: {servoX} y: {servoY}\033[0m')  # Print in blue color

        else:
            target_status = "Target: No Hand Detected"

        # Draw the Terminator-style overlay
        draw_terminator_overlay(img, servoX, servoY, target_status)

        # Display the image
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    except urllib.error.URLError as e:
        error_message = f"Error fetching image: {e.reason}"
        print(error_message)
        print(f"\033[92mTrying to reconnect...\033[0m")
        # Draw the error message on the frame
        img = np.zeros((hs, ws, 3), np.uint8)  # Black frame
        draw_terminator_overlay(img, prev_servoX, prev_servoY, target_status, error_message=error_message)
        cv2.imshow("Image", img)
        cv2.waitKey(5000)  # Display the error message for 5 seconds
        continue

    except Exception as e:
        error_message = f"Error: {e}"
        print(error_message)
        print(f"\033[92mTrying to reconnect...\033[0m")
        # Draw the error message on the frame
        img = np.zeros((hs, ws, 3), np.uint8)  # Black frame
        draw_terminator_overlay(img, prev_servoX, prev_servoY, target_status, error_message=error_message)
        cv2.imshow("Image", img)
        cv2.waitKey(5000)  # Display the error message for 5 seconds
        continue

cv2.destroyAllWindows()
