import cv2
import dlib
import numpy as np
import time
import pyttsx3
from scipy.spatial import distance as dist

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Load models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Eye and face landmarks
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))
EAR_THRESHOLD = 0.25
YAW_THRESHOLD = 20  # Degrees threshold for head turning
LOOK_AWAY_FRAMES = 15  # Reduced for quicker feedback

# 3D model points (approximate)
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (30)
    (0.0, -330.0, -65.0),        # Chin (8)
    (-225.0, 170.0, -135.0),     # Left eye left corner (36)
    (225.0, 170.0, -135.0),      # Right eye right corner (45)
    (-150.0, -150.0, -125.0),    # Left Mouth corner (48)
    (150.0, -150.0, -125.0)      # Right mouth corner (54)
], dtype=np.float64)

# Variables
attention_score = 100
blinks = 0
last_blink_time = time.time()
look_away_counter = 0
frame_history = []

# EAR Calculation
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Convert rotation matrix to Euler angles
def rotation_matrix_to_euler_angles(R):
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
    return np.degrees(x), np.degrees(y), np.degrees(z)

# Start capturing
cap = cv2.VideoCapture(0)
speak("Hello! Welcome to your study session.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Initialize camera matrix each frame (adjust if resolution changes)
    height, width = frame.shape[:2]
    focal_length = width
    center = (width/2, height/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4,1))  # Assuming no lens distortion
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) == 0:
        attention_score -= 2  # Reduce score if no face detected
        look_away_counter += 1
    else:
        for face in faces:
            landmarks = predictor(gray, face)
            
            # Eye tracking
            left_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in LEFT_EYE])
            right_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in RIGHT_EYE])
            avg_EAR = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
            
            # Blink detection
            if avg_EAR < EAR_THRESHOLD:
                if time.time() - last_blink_time > 0.2:
                    blinks += 1
                    last_blink_time = time.time()
                    attention_score -= 1
            else:
                attention_score += 0.5  # Slow recovery
            
            # Head pose estimation
            image_points = np.array([
                (landmarks.part(30).x, landmarks.part(30).y),
                (landmarks.part(8).x, landmarks.part(8).y),
                (landmarks.part(36).x, landmarks.part(36).y),
                (landmarks.part(45).x, landmarks.part(45).y),
                (landmarks.part(48).x, landmarks.part(48).y),
                (landmarks.part(54).x, landmarks.part(54).y)
            ], dtype=np.float64)

            # Solve PnP to get head pose
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, 
                image_points, 
                camera_matrix, 
                dist_coeffs, 
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            # Get Euler angles
            rotation_mat, _ = cv2.Rodrigues(rotation_vector)
            pitch, yaw, roll = rotation_matrix_to_euler_angles(rotation_mat)
            
            # Check for head turning
            if abs(yaw) > YAW_THRESHOLD:
                attention_score -= 3
                look_away_counter += 1
            else:
                look_away_counter = max(look_away_counter - 1, 0)
            
            # Visual feedback
            cv2.polylines(frame, [left_eye, right_eye], isClosed=True, color=(0, 255, 0), thickness=1)
            cv2.putText(frame, f"Yaw: {int(yaw)}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Alert if looking away too long
    if look_away_counter >= LOOK_AWAY_FRAMES:
        speak("Pay attention!")
        look_away_counter = 0  # Reset counter after alert
    
    # Keep score in range
    attention_score = max(0, min(100, attention_score))
    frame_history.append(attention_score)
    
    # Display score
    cv2.putText(frame, f"Attention Score: {int(attention_score)}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Real-Time Attention Tracker", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Display final stats
average_score = sum(frame_history) / len(frame_history)
speak(f"Your study session has ended. Your average attention score was {int(average_score)}")
print(f"Average Attention Score: {int(average_score)}")

cap.release()
cv2.destroyAllWindows()