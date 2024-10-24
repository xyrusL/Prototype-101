# Alphanumeric Recognition System

## Overview
This project is an advanced Alphanumeric Recognition System developed by CS San Mateo 2024. It utilizes computer vision and machine learning techniques to recognize hand gestures, perform speech recognition, and offer various interactive features.

## Features
- Hand gesture recognition for both left and right hands
- Face and body detection
- Speech recognition and text-to-speech functionality
- Real-time word and sentence formation from recognized gestures
- Customizable settings through an interactive menu
- FPS display and performance metrics

## Requirements
- Python 3.x
- OpenCV
- MediaPipe
- TensorFlow
- PyGame
- SpeechRecognition
- pyttsx3
- Tkinter

## Installation
1. Clone the repository: git clone https://github.com/your-username/alphanumeric-recognition.git
2. Install the required packages: pip install -r requirements.txt

## Usage
Run the main application:

### Controls
- 'Esc' or 'q': Exit the application
- 'd': Toggle word/sentence mode
- 'f': Delete last character in word mode
- 's': Toggle speech recognition
- 'a': Toggle automatic listening (in speech recognition mode)
- 'l': Start/stop listening (in manual speech recognition mode)
- 'r': Read aloud the current word/sentence or recognized speech
- 'm': Open settings menu

## Configuration
The system can be configured through the settings menu ('m' key) or by modifying the `config.py` file. Adjustable parameters include:
- Camera settings
- Detection confidence thresholds
- Visual display options
- Voice version for sound feedback

## Models
The system uses TensorFlow Lite models for hand gesture classification:
- `rightHandModel.tflite`: For right-hand gestures
- `leftHandModel.tflite`: For left-hand gestures

## Contributing
Contributions to improve the system are welcome. Please follow these steps:
1. Fork the repository
2. Create a new branch
3. Make your changes and commit them
4. Push to your fork and submit a pull request
