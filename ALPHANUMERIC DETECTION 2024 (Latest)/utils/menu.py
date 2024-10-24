import tkinter as tk
from collections import namedtuple
from utils.config import (
    showLandMarks, soundDetected, letter_confidence_time, voice_version, 
    show_face_detection, show_body_detection, camera_id, show_terminal_output,
    cap_width, cap_height, min_detection_confidence, min_tracking_confidence, 
    background_color, background_color_option, hand_detected_color_bg,
    hand_detected_color_option
)

def print_global_values():
    global show_face_detection, show_body_detection, showLandMarks
    global voice_version, min_detection_confidence, min_tracking_confidence
    global cap_width, cap_height, camera_id
    global background_color, hand_detected_color_bg

    print(f"Show Face Detection: {show_face_detection}")
    print(f"Show Body Detection: {show_body_detection}")
    print(f"Show Detection For Hand: {showLandMarks}")
    print(f"Sound Detected: {soundDetected}")
    print(f"Voice Version: {voice_version}")
    print(f"Min Detection Confidence: {min_detection_confidence}")
    print(f"Min Tracking Confidence: {min_tracking_confidence}")
    print(f"Letter Confidence Time: {letter_confidence_time}")
    print(f"Cap Width: {cap_width}")
    print(f"Cap Height: {cap_height}")
    print(f"Camera ID: {camera_id}")
    print(f"Background Color: {background_color}")
    print(f"Hand Detected Color: {hand_detected_color_bg}")

GlobalValues = namedtuple('GlobalValues', [
    'show_face_detection',
    'show_body_detection',
    'showLandMarks',
    'soundDetected',
    'voice_version',
    'min_detection_confidence',
    'min_tracking_confidence',
    'letter_confidence_time',
    'cap_width',
    'cap_height',
    'camera_id',
    'background_color',
    'hand_detected_color_bg'
])

def return_global_values():
    return GlobalValues(
        show_face_detection=show_face_detection,
        show_body_detection=show_body_detection,
        showLandMarks=showLandMarks,
        soundDetected=soundDetected,
        voice_version=voice_version,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        letter_confidence_time=letter_confidence_time,
        cap_width=cap_width,
        cap_height=cap_height,
        camera_id=camera_id,
        background_color=background_color,
        hand_detected_color_bg=hand_detected_color_bg
    )

class MenuInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Menu Interface")
        self.root.geometry("400x550")  
        self.root.configure(bg="#f0f0f0")  

        # Boolean variables
        self.show_face_detection = tk.BooleanVar(value=show_face_detection)
        self.show_body_detection = tk.BooleanVar(value=show_body_detection)
        self.showLandMarks = tk.BooleanVar(value=showLandMarks)
        self.soundDetected = tk.BooleanVar(value=soundDetected)
        self.show_terminal_output = tk.BooleanVar(value=show_terminal_output)

        # Integer variable for Voice Version and Camera ID
        self.voice_version = tk.IntVar(value=voice_version)
        self.camera_id = tk.IntVar(value=camera_id)

        # Float variables for detection thresholds
        self.min_detection_confidence = tk.DoubleVar(value=min_detection_confidence)
        self.min_tracking_confidence = tk.DoubleVar(value=min_tracking_confidence)
        self.letter_confidence_time = tk.DoubleVar(value=letter_confidence_time)

        # Integer variables for Frame dimensions
        self.cap_width = tk.IntVar(value=cap_width)
        self.cap_height = tk.IntVar(value=cap_height)

        # Background color option variable
        self.background_color_option = tk.StringVar(value=background_color_option) 
        self.hand_detected_color_option = tk.StringVar(value=hand_detected_color_option)

        # Creating checkbuttons for boolean options
        self.create_checkbutton("Show Face Detection", self.show_face_detection)
        self.create_checkbutton("Show Body Detection", self.show_body_detection)
        self.create_checkbutton("Show Detection For Hand", self.showLandMarks)
        self.create_checkbutton("System Sound", self.soundDetected)
        self.create_checkbutton("Global Values (For debug)", self.show_terminal_output)

        # Option menu for voice version
        self.create_option_menu("Voice Version", self.voice_version, [1, 2])

        # Entry fields for detection thresholds
        self.create_entry("Min Detection Confidence", self.min_detection_confidence)
        self.create_entry("Min Tracking Confidence", self.min_tracking_confidence)
        self.create_entry("Letter Confidence Time", self.letter_confidence_time)

        # Entry fields for frame dimensions
        self.create_entry("Frame Width", self.cap_width)
        self.create_entry("Frame Height", self.cap_height)

        # Entry field for Camera ID
        self.create_entry("Camera ID", self.camera_id)

        # Option menu for background color
        self.create_option_menu("Background Color", self.background_color_option, ["grey", "black"])  
        self.create_option_menu("Hand Detected Color", self.hand_detected_color_option, ["default", "magenta", "darkgreen", "purple", "darkred",  "maroon5"])

        # Bind the "M" key to close the tkinter window
        self.root.bind("<Key-m>", self.close_window)

        # Set focus to the root window to ensure key bindings work
        self.root.focus_force()

    def create_checkbutton(self, text, variable):
        checkbutton = tk.Checkbutton(
            self.root, text=text, variable=variable,
            onvalue=True, offvalue=False, bg="#f0f0f0",
            fg="#000000",  # Black text for contrast
            font=("Verdana", 12),
            command=lambda: self.update_global_variable(text, variable.get())
        )
        checkbutton.pack(anchor="w", pady=5, padx=10)

    def create_option_menu(self, text, variable, options):
        frame = tk.Frame(self.root, bg="#f0f0f0")
        frame.pack(anchor="w", pady=5, padx=10)
    
        label = tk.Label(frame, text=text, bg="#f0f0f0", fg="#000000", font=("Verdana", 12))
        label.pack(side="left")
    
        option_menu = tk.OptionMenu(
            frame, variable, *options,
            command=lambda _: self.update_global_variable(text, variable.get())
        )
        option_menu.configure(bg="#ffffff", fg="#000000", font=("Verdana", 12))  
        option_menu.pack(side="left")
        variable.trace_add("write", lambda *args: option_menu.set_menu(variable.get(), *options))


    def create_entry(self, text, variable):
        frame = tk.Frame(self.root, bg="#f0f0f0")
        frame.pack(anchor="w", pady=5, padx=10)

        label = tk.Label(frame, text=text, bg="#f0f0f0", fg="#000000", font=("Verdana", 12))
        label.pack(side="left")

        entry = tk.Entry(frame, textvariable=variable, bg="#ffffff", fg="#000000", font=("Verdana", 12), width=10)
        entry.pack(side="left")
        
        # Automatically save the value when it changes
        variable.trace_add("write", lambda *args: self.update_global_variable(text, variable.get()))

    def update_global_variable(self, option, value):
        global show_face_detection, show_body_detection, showLandMarks, soundDetected
        global voice_version, min_detection_confidence, min_tracking_confidence, letter_confidence_time
        global cap_width, cap_height, camera_id, background_color, background_color_option
        global show_terminal_output, hand_detected_color_bg, hand_detected_color_option

        if option == "Show Face Detection":
            show_face_detection = value
        elif option == "Show Body Detection":
            show_body_detection = value
        elif option == "Show Detection For Hand":
            showLandMarks = value
        elif option == "System Sound":
            soundDetected = value
        elif option == "Global Values (For debug)":
            show_terminal_output = value
        elif option == "Voice Version":
            voice_version = value
        elif option == "Min Detection Confidence":
            min_detection_confidence = value
        elif option == "Min Tracking Confidence":
            min_tracking_confidence = value
        elif option == "Letter Confidence Time":
            letter_confidence_time = value
        elif option == "Frame Width":
            cap_width = value
        elif option == "Frame Height":
            cap_height = value
        elif option == "Camera ID":
            camera_id = value
        elif option == "Background Color":
            background_color_option = value
            background_color = (50, 50, 50) if background_color_option == "grey" else (25, 25, 25) 
        elif option == "Hand Detected Color":
            hand_detected_color_option = value
            color_map = {
                "default": (139, 0, 0),  
                "magenta": (139, 0, 139),
                "darkgreen": ( 0, 100, 0),
                "purple": (75, 0, 130),
                "darkred": (0, 0, 139),
                "maroon5": (139, 28, 98)
            }
            hand_detected_color_bg = color_map.get(hand_detected_color_option, (139, 0, 0))

        if show_terminal_output:
            print_global_values()

    def close_window(self, event):
        """Close the tkinter window when the 'M' key is pressed."""
        self.root.destroy()
