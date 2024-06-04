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
url = "http://192.168.254.137/480x320.jpg"
ws, hs = 480, 320

# Initialize Arduino board and servo pins
port = "COM3"
board = pyfirmata.Arduino(port)
servo_pinX = board.get_pin('d:9:s')  # pin 9 Arduino
servo_pinY = board.get_pin('d:10:s')  # pin 10 Arduino

# Variables to store previous servo positions for smooth movement
prev_servoX = 90
prev_servoY = 90

# Center of the frame
center_x = ws // 2
center_y = hs // 2

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
            # Hand Visualization
            for lm in multiHandDetection:
                mpDraw.draw_landmarks(img, lm, mpHand.HAND_CONNECTIONS,
                                      mpDraw.DrawingSpec(color=(0, 255, 255), thickness=4, circle_radius=7),
                                      mpDraw.DrawingSpec(color=(0, 0, 0), thickness=4))

            # Hand Tracking
            singleHandDetection = multiHandDetection[0]
            for lm in singleHandDetection.landmark:
                h, w, c = img.shape
                lm_x, lm_y = int(lm.x * w), int(lm.y * h)
                lmList.append([lm_x, lm_y])

            # Focus on index finger tip (landmark 8)
            index_tip = lmList[8]
            px, py = index_tip[0], index_tip[1]
            cv2.circle(img, (px, py), 15, (255, 0, 255), cv2.FILLED)
            cv2.putText(img, str((px, py)), (px + 10, py - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

            # Calculate the offset from the center
            offset_x = center_x - px
            offset_y = center_y - py

            # Convert position to degree value, adjusting for center offset
            servoX = int(np.interp(px, [0, ws], [0, 180]))
            servoY = int(np.interp(py, [0, hs], [0, 180]))

            # Smooth movement
            smooth_factor = 0.2
            servoX = int(prev_servoX + smooth_factor * (servoX - prev_servoX))
            servoY = int(prev_servoY + smooth_factor * (servoY - prev_servoY))
            prev_servoX, prev_servoY = servoX, servoY

            cv2.rectangle(img, (40, 20), (350, 110), (0, 255, 255), cv2.FILLED)
            cv2.putText(img, f'Servo X: {servoX} deg', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.putText(img, f'Servo Y: {servoY} deg', (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

            # Write to servo pins
            servo_pinX.write(servoX)
            servo_pinY.write(servoY)

            print(f'\033[92mHand Position x: {px} y: {py}\033[0m')  # Print in green color
            print(f'\033[92mServo Value x: {servoX} y: {servoY}\033[0m')  # Print in green color

        # Draw the crosshair at the center
        cv2.line(img, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 2)
        cv2.line(img, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 2)

        # Display the image
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    except urllib.error.URLError as e:
        print(f"Error fetching image: {e.reason}")
        print(f"\033[92mTrying to reconnect...\033[0m")
        continue  # Skip the rest of the loop iteration
    except Exception as e:
        print(f"An error occurred: {e}")
        break  # Exit the loop if an unhandled exception occurs

# Release resources
cv2.destroyAllWindows()