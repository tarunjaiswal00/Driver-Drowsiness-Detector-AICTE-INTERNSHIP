import cv2
import dlib
import time
from scipy.spatial import distance as dist
from imutils import face_utils
from ultralytics import YOLO  

print("Loading models...")

detector_dlib = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
except RuntimeError as e:
    print("\n[ERROR] Missing 'shape_predictor_68_face_landmarks.dat' file.")
    print("Download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    print("And unzip it in the same folder as this script.\n")
    exit()
EAR_THRESHOLD = 0.25
CONSECUTIVE_FRAMES = 30
COUNTER = 0

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

model_yolo = YOLO("yolov8n.pt")
class_names = model_yolo.names

print("Starting webcam... (Press 'q' to quit)")
cap = cv2.VideoCapture(0)
time.sleep(2.0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector_dlib(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        if ear < EAR_THRESHOLD:
            COUNTER += 1
            if COUNTER >= CONSECUTIVE_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    results = model_yolo(frame, stream=True, verbose=False)

    for r in results:
        for box in r.boxes:
            
            cls = int(box.cls[0])
         
            if class_names[cls] == "cell phone":
         
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
    
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                
    
                cv2.putText(frame, "DISTRACTION ALERT!", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Driver Monitor (FINAL)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Closing application...")
cap.release()
cv2.destroyAllWindows()