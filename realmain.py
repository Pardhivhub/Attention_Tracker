import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import time

# Load models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Indices for eyes
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

# Thresholds
EAR_THRESHOLD = 0.25
ANGLE_THRESHOLD = 15
MISSING_FACE_PENALTY = 5

# Score tracking
attention_score = 100
blinks = 0
last_blink_time = time.time()
score_log = []  # Stores scores per second

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Function to calculate face angle
def get_face_angle(landmarks):
    left_eye = (landmarks.part(36).x, landmarks.part(36).y)
    right_eye = (landmarks.part(45).x, landmarks.part(45).y)
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    return abs(angle)

# Start capturing
cap = cv2.VideoCapture(0)
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    current_time = time.time()
    face_detected = False
    
    for face in faces:
        face_detected = True
        landmarks = predictor(gray, face)
        left_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in LEFT_EYE])
        right_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in RIGHT_EYE])
        
        # Compute EAR
        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)
        avg_EAR = (left_EAR + right_EAR) / 2.0
        
        # Compute face angle
        angle = get_face_angle(landmarks)
        
        # Blink detection
        if avg_EAR < EAR_THRESHOLD:
            if time.time() - last_blink_time > 0.2:
                blinks += 1
                last_blink_time = time.time()
                attention_score -= 2
        else:
            attention_score += 0.5  # Slow recovery
        
        # Face angle check
        if angle > ANGLE_THRESHOLD:
            attention_score -= 3
            cv2.putText(frame, "LOOKING AWAY!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Draw eye landmarks
        cv2.polylines(frame, [left_eye], isClosed=True, color=(0, 255, 0), thickness=1)
        cv2.polylines(frame, [right_eye], isClosed=True, color=(0, 255, 0), thickness=1)
    
    if not face_detected:
        attention_score -= MISSING_FACE_PENALTY  # Penalize missing face
        cv2.putText(frame, "NO FACE DETECTED!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Keep score in range 0-100
    attention_score = max(0, min(100, attention_score))
    
    # Log score every second
    if current_time - start_time >= 1:
        score_log.append(attention_score)
        start_time = current_time
    
    # Display Attention Score
    cv2.putText(frame, f"Attention Score: {int(attention_score)}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    cv2.imshow("Real-Time Attention Tracker", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# Print session summary
print("\n=== SESSION SUMMARY ===")
print(f"Total Time Tracked: {len(score_log)} seconds")
print(f"Final Attention Score: {int(attention_score)}")
print(f"Total Blinks: {blinks}")
print(f"Average Score: {sum(score_log) / len(score_log) if score_log else 100:.2f}")

cap.release()
cv2.destroyAllWindows()

