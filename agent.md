1. Project context, goals, and specifics:
    A user will be able to manipulate their computer’s cursor without using the physical mouse and keyboard. Using eye movement they will be able to move the cursor on screen and using head movements they can initiate clicks and double clicks. 

    - On startup, the user will calibrate the app by performing various tests.
        - We need to know the resolution of the monitor
        - Find the coordinate ranges by prompting the user to look at the 4 corners of the screen
        - Find the resting head position and set that position to 0°. If it deviates beyond a threshold (like 15°) initiate the corresponding click.
        - Since webcam position and the flatness of a monitor can make eye movements seem non linear, record the eye position when the user is looking at the exact middle of their screen.

    EYE

    - A live video feed will be fed into an eye tracking tool to determine where on the screen the user is looking at.
        - track the absolute centers of the left and right pupils, then find the midpoint between the two eyes to calculate the coordinate and feed it into a smoothing algorithm.
        - if eye is closed for a specified amount of frames ⇒ NO CLICK
        - if there eyes are past the bounds of the screen ⇒ NO CLICK
        - AND if the head roll is beyond a certain point ⇒ NO click
        - If the calibration json file is not occupied or is missing information, begin the calibration tests or if the user wishes to redo the calibration.

    - The app will use those coordinates to move the cursor to the corresponding spot on the screen.
        - Use a smoothing algorithm to reduce jittering of the cursor. (Consider Kalman filtering or normal coordinate averaging.)

    HEAD

    - The tool will also track head movements from the user, allowing certain head tilts to indicate a click or a double click.
        - Using Media pipe to track coordinates on the face.
            - feed the coordinates and a numpy array (to represent a generic 3d face) into solvePnP. It will output the Pitch, yaw, and roll values of the head.
            - 2 Matrices 3x1 and 3x3 translating into angles
            - ( x , y , z)

    - When a click or double click is initiated, the app interrupts the movement of the cursor and executes the corresponding click.

2. Tech Stack Rules:
    - Computer vision: use media pipe for facial landmarks and opencv-python for video capture. do not use dlib or Haar cascades.
    - OS Automation: Use pynput for mouse control.
    - UI: Use customtkinter for calibration screen.

3. Architectural mandates:
    - Must use multithreading using python's threading and queue modules. 
    - Maintain strict separation of concerns:
        1. Thread 1: camera capture (read the frames to a queue)
        2. Thread 2: Vision processing (reads frames, runs Media Pipe, calculate math, outputs coordinates to a queue).
        3. Thread 3: Mouse controller (reads coordinates and moves the cursor.)

4. Domain Specific Knowledge:
    - Pupils: Use mediaPipe face mesh landmarks 468 (left pupil center) and 473 (right pupil center)
    - Head tracking: use cv2.solvePnP with 6 anchor points (nose tip, chin, outer eyes, mouth corners) to calculate pitch and roll.
    - Mapping: Cursor movement must use a 4 quadrant linear interpolation mapping based on a center anchor point to correct for webcam offset.

5. General Rules:
    - write bare minimum comments. 