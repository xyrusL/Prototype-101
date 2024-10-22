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
ws, hs = 800, 600 # Width and Height of the frame

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

# Flag to check if it's the first detection
first_detection = True

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
            for handLms in multiHandDetection:
                mpDraw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONS)

                # Get the bounding box for the hand
                bbox = cv2.boundingRect(np.array([[lm.x * ws, lm.y * hs] for lm in handLms.landmark]).astype(int))
                x, y, w, h = bbox
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Calculate center of the bounding box
                bbox_center_x = x + w // 2
                bbox_center_y = y + h // 2

            # Calculate the offset from the center of the bounding box to the center of the frame
            offset_x = center_x - bbox_center_x
            offset_y = center_y - bbox_center_y

            # Check if the bounding box is outside the frame
            if bbox_center_x < 0 or bbox_center_x > ws or bbox_center_y < 0 or bbox_center_y > hs:
                # Trigger Algorithm: Move the servo to catch the hand
                if bbox_center_x < 0:
                    offset_x = center_x
                elif bbox_center_x > ws:
                    offset_x = -center_x
                if bbox_center_y < 0:
                    offset_y = center_y
                elif bbox_center_y > hs:
                    offset_y = -center_y

            # Convert position to degree value, adjusting for center offset
            servoX = int(np.interp(center_x - offset_x, [0, ws], [0, 180]))
            servoY = int(np.interp(center_y - offset_y, [0, hs], [0, 180]))

            # Smooth movement
            smooth_factor = 0.2
            if first_detection:
                # Gradually move to the first detected position
                steps = 10
                stepX = (servoX - prev_servoX) / steps
                stepY = (servoY - prev_servoY) / steps
                for i in range(steps):
                    prev_servoX += stepX
                    prev_servoY += stepY
                    servo_pinX.write(int(prev_servoX))
                    servo_pinY.write(int(prev_servoY))
                    cv2.waitKey(10)  # Small delay to simulate smooth movement
                first_detection = False
            else:
                servoX = int(prev_servoX + smooth_factor * (servoX - prev_servoX))
                servoY = int(prev_servoY + smooth_factor * (servoY - prev_servoY))
                prev_servoX, prev_servoY = servoX, servoY

            cv2.rectangle(img, (40, 20), (350, 110), (0, 255, 255), cv2.FILLED)
            cv2.putText(img, f'Servo X: {servoX} deg', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.putText(img, f'Servo Y: {servoY} deg', (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

            # Write to servo pins
            servo_pinX.write(servoX)
            servo_pinY.write(servoY)

            print(f'\033[92mHand Position x: {bbox_center_x} y: {bbox_center_y}\033[0m')  # Print in green color
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
        print(f"\033[92mTrying to reconnect...\033[0m")
        continue  # Skip the rest of the loop iteration

# Release resources
cv2.destroyAllWindows()
