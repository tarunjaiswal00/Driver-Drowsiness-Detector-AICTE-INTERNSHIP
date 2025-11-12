# Driver Drowsiness & Distraction Detector (AICTE Internship)
**Intern:** Tarun Jaiswal
**Project:** AICTE Cycle 4 Internship (Oct-Nov 2025) - Automotive Theme

This is the final project submission for the "Automotive - Object Detection" theme. This project is a real-time monitoring system that uses a hybrid computer vision approach to detect both driver drowsiness and distraction.

---

## Features
1.  **Drowsiness Detection:**
    * Uses **Dlib** to perform real-time facial landmark detection.
    * Calculates the **Eye Aspect Ratio (EAR)** to numerically quantify eye opening.
    * Triggers a "DROWSINESS ALERT!" if the EAR drops below a set threshold for a specific duration (detecting a micro-sleep).
2.  **Distraction Detection:**
    * Uses a pre-trained **YOLOv8** object detection model.
    * Scans the video feed for a "cell phone".
    * Triggers a "DISTRACTION ALERT!" with a bounding box if a phone is detected.

---

## How to Run This Project
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/tarunjaiswal00/Driver-Drowsiness-Detector-AICTE-INTERNSHIP.git](https://github.com/tarunjaiswal00/Driver-Drowsiness-Detector-AICTE-INTERNSHIP.git)
    cd Driver-Drowsiness-Detector-AICTE-INTERNSHIP
    ```
2.  **Download the Dlib Model:**
    The `shape_predictor_68_face_landmarks.dat` file is required but not included in the repo (due to its size).
    * **Download Link:** [http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
    * Unzip the file and place `shape_predictor_68_face_landmarks.dat` in the same project folder as `detector.py`.

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the application:**
    ```bash
    python detector.py
    ```