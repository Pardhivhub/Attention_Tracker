import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import time
import math

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Constants for focus tracking
EAR_CHANGE_THRESHOLD = 0.05   # Threshold for detecting blinks
MIN_BLINK_DURATION = 0.05     # Minimum duration of a blink in seconds
MAX_BLINK_DURATION = 0.4      # Maximum duration of a blink in seconds
COOLDOWN_PERIOD = 0.2         # Minimum time between blinks
BLINK_OPTIMAL_RANGE = (8, 12) # Optimal blinks per minute range
OPTIMAL_EAR = 0.25            # Optimal EAR value
HEAD_MOVEMENT_THRESHOLD = 40  # Head movement threshold (in degrees)

# EAR evaluation thresholds
EAR_THRESHOLDS = [
    (0.35, "Excellent", (0, 180, 0)),     # Green
    (0.30, "Very Good", (60, 180, 0)),    # Light green
    (0.25, "Good", (120, 180, 0)),        # Yellow-green
    (0.20, "Tired", (180, 180, 0)),       # Yellow
    (0.18, "Drowsy", (180, 120, 0)),      # Orange
    (0.10, "Very Drowsy", (180, 60, 0)),  # Red-orange
    (0.00, "Sleeping", (180, 0, 0))       # Red
]

def calculate_ear(eye):
    """Calculate eye aspect ratio"""
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def shape_to_np(shape, dtype="int"):
    """Convert dlib shape to numpy array"""
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def rotation_matrix_to_euler_angles(R):
    """Convert rotation matrix to Euler angles (radians)"""
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

def evaluate_ear(ear):
    """Evaluate EAR value and return status and color"""
    for threshold, status, color in EAR_THRESHOLDS:
        if ear >= threshold:
            return status, color
    return "Sleeping", (180, 0, 0)

class HeadPoseEstimator:
    """Class to handle head pose estimation with calibration"""
    def __init__(self):
        # 3D model points for a generic face
        self.model_points = np.array([
            (0.0, 0.0, 0.0),            # Nose tip
            (0.0, -330.0, -65.0),       # Chin
            (-225.0, 170.0, -135.0),    # Left eye left corner
            (225.0, 170.0, -135.0),     # Right eye right corner
            (-150.0, -150.0, -125.0),   # Left mouth corner
            (150.0, -150.0, -125.0),    # Right mouth corner
            (0.0, 70.0, -40.0),         # Nose bridge 
            (-330.0, 100.0, -30.0),     # Jaw left
            (330.0, 100.0, -30.0)       # Jaw right
        ])
        
        # Calibration variables
        self.is_calibrated = False
        self.roll_offset = 0
        self.pitch_offset = 0
        self.yaw_offset = 0
        
        # For smoothing
        self.prev_angles = None
        
    def calibrate(self, angles):
        """Calibrate using the current pose as neutral"""
        if angles[0] is not None:
            self.roll_offset = angles[0]
            self.pitch_offset = angles[1]
            self.yaw_offset = angles[2]
            self.is_calibrated = True
            print(f"Calibration completed: Roll={self.roll_offset:.2f}, "
                  f"Pitch={self.pitch_offset:.2f}, Yaw={self.yaw_offset:.2f}")
            return True
        return False
        
    def get_pose(self, shape, frame_size):
        """Get the head pose angles"""
        # 2D image points from facial landmarks
        image_points = np.array([
            tuple(shape[30]),  # Nose tip
            tuple(shape[8]),   # Chin
            tuple(shape[36]),  # Left eye left corner
            tuple(shape[45]),  # Right eye right corner
            tuple(shape[48]),  # Left mouth corner
            tuple(shape[54]),  # Right mouth corner
            tuple(shape[33]),  # Nose bridge
            tuple(shape[0]),   # Jaw left
            tuple(shape[16])   # Jaw right
        ], dtype='double')
        
        # Camera parameters
        focal_length = frame_size[1]
        center = (frame_size[1] / 2, frame_size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        
        # Assuming no lens distortion
        dist_coeffs = np.zeros((4, 1))
        
        # Solve for pose
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, image_points, camera_matrix, dist_coeffs, 
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None, None, None
        
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Get Euler angles in degrees
        euler_angles = rotation_matrix_to_euler_angles(rotation_matrix)
        pitch, yaw, roll = [np.degrees(angle) for angle in euler_angles]
        
        # Apply smoothing
        if self.prev_angles is not None:
            alpha = 0.3  # Smoothing factor
            roll = alpha * roll + (1 - alpha) * self.prev_angles[0]
            pitch = alpha * pitch + (1 - alpha) * self.prev_angles[1]
            yaw = alpha * yaw + (1 - alpha) * self.prev_angles[2]
        
        self.prev_angles = (roll, pitch, yaw)
        
        # Apply calibration if calibrated
        if self.is_calibrated:
            roll -= self.roll_offset
            pitch -= self.pitch_offset
            yaw -= self.yaw_offset
        
        return roll, pitch, yaw

def compute_focus_score(ear_value, blink_rate, head_movement, baseline_ear):
    """Compute a comprehensive focus score based on multiple metrics"""
    # Initialize component scores
    ear_score = 0
    blink_score = 0
    head_score = 0
    
    # EAR component (0-100)
    if ear_value >= 0.35:
        ear_score = 100  # Excellent
    elif ear_value >= 0.30:
        ear_score = 90   # Very good
    elif ear_value >= 0.25:
        ear_score = 80   # Good
    elif ear_value >= 0.20:
        ear_score = 60   # Tired
    elif ear_value >= 0.18:
        ear_score = 40   # Drowsy
    elif ear_value >= 0.10:
        ear_score = 20   # Very drowsy
    else:
        ear_score = 0    # Sleeping
    
    # Blink rate component (0-100)
    if blink_rate >= 0:  # Ensure we have a valid blink rate
        if BLINK_OPTIMAL_RANGE[0] <= blink_rate <= BLINK_OPTIMAL_RANGE[1]:
            # Optimal range
            blink_score = 100
        elif blink_rate < BLINK_OPTIMAL_RANGE[0]:
            # Too few blinks - may indicate staring/fixation
            deficit = BLINK_OPTIMAL_RANGE[0] - blink_rate
            blink_score = max(0, 100 - (deficit * 10))  # Lose 10 points per blink below optimal
        else:
            # Too many blinks - may indicate distraction/anxiety
            excess = blink_rate - BLINK_OPTIMAL_RANGE[1]
            blink_score = max(0, 100 - (excess * 6))    # Lose 6 points per blink above optimal
    
    # Head movement component (0-100)
    if head_movement <= HEAD_MOVEMENT_THRESHOLD / 3:
        head_score = 100  # Minimal movement
    elif head_movement <= HEAD_MOVEMENT_THRESHOLD * 2/3:
        head_score = 80   # Moderate movement
    elif head_movement <= HEAD_MOVEMENT_THRESHOLD:
        head_score = 60   # Significant movement
    elif head_movement <= HEAD_MOVEMENT_THRESHOLD * 4/3:
        head_score = 40   # Excessive movement
    elif head_movement <= HEAD_MOVEMENT_THRESHOLD * 5/3:
        head_score = 20   # Very excessive movement
    else:
        head_score = 0    # Extreme movement
    
    # Weighted combination (EAR 30%, Blink 50%, Head 20%)
    final_score = int(0.3 * ear_score + 0.5 * blink_score + 0.2 * head_score)
    
    return final_score

def get_focus_status(score):
    """Get focus status description based on score"""
    if score >= 90:
        return "Excellent Focus", (0, 180, 0)  # Green
    elif score >= 80:
        return "Very Good Focus", (60, 180, 0)  # Light green
    elif score >= 70:
        return "Good Focus", (120, 180, 0)  # Yellow-green
    elif score >= 60:
        return "Moderate Focus", (180, 180, 0)  # Yellow
    elif score >= 40:
        return "Low Focus", (180, 120, 0)  # Orange
    elif score >= 20:
        return "Very Low Focus", (180, 60, 0)  # Red-orange
    else:
        return "Extremely Low Focus", (180, 0, 0)  # Red

def draw_focus_gauge(img, score, x, y, width, height):
    """Draw a professional-looking focus gauge"""
    # Draw background
    cv2.rectangle(img, (x, y), (x + width, y + height), (60, 60, 60), -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), (180, 180, 180), 2)
    
    # Draw colored segments
    segment_width = width - 4
    segment_height = height - 4
    segment_x = x + 2
    segment_y = y + 2
    
    # Calculate fill amount
    fill_width = int(segment_width * (score / 100))
    
    # Get color based on score
    _, color = get_focus_status(score)
    
    # Draw fill
    cv2.rectangle(img, (segment_x, segment_y), 
                 (segment_x + fill_width, segment_y + segment_height), 
                 color, -1)
    
    # Draw markers
    for i in range(0, 101, 10):
        mark_x = segment_x + int(segment_width * (i / 100))
        mark_height = 8 if i % 20 == 0 else 5
        cv2.line(img, (mark_x, segment_y), 
                (mark_x, segment_y + mark_height), 
                (220, 220, 220), 1)
        
        # Add label for major markers
        if i % 20 == 0:
            cv2.putText(img, str(i), (mark_x - 7, segment_y + segment_height + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
    
    # Display score
    cv2.putText(img, f"{score}", (x + width//2 - 15, y + height//2 + 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

def draw_metric_box(img, title, value, status, color, x, y, width, height):
    """Draw a professional-looking metric box"""
    # Draw background
    cv2.rectangle(img, (x, y), (x + width, y + height), (60, 60, 60), -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), (180, 180, 180), 2)
    
    # Draw header
    cv2.rectangle(img, (x, y), (x + width, y + 30), (40, 40, 40), -1)
    cv2.putText(img, title, (x + 10, y + 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
    
    # Draw value
    cv2.putText(img, f"{value}", (x + width//2 - 30, y + 65), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    # Draw status with color
    cv2.putText(img, status, (x + width//2 - len(status)*4, y + 95), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def draw_head_pose_visualization(img, roll, pitch, yaw, x, y, size=50):
    """Draw a 3D arrow representing head orientation"""
    # Draw coordinate frame
    origin = (x, y)
    
    # Calculate arrow endpoints using trigonometry
    roll_rad = math.radians(roll)
    pitch_rad = math.radians(pitch)
    yaw_rad = math.radians(yaw)
    
    # X axis (roll) - red
    x_end = (int(x + size * math.cos(yaw_rad)), 
             int(y + size * math.sin(yaw_rad) * math.sin(roll_rad)))
    cv2.arrowedLine(img, origin, x_end, (0, 0, 255), 2)
    
    # Y axis (pitch) - green
    y_end = (int(x + size * math.sin(yaw_rad) * math.sin(pitch_rad)), 
             int(y + size * math.cos(pitch_rad)))
    cv2.arrowedLine(img, origin, y_end, (0, 255, 0), 2)
    
    # Z axis (yaw) - blue
    z_end = (int(x + size * math.sin(yaw_rad)), 
             int(y + size * math.cos(yaw_rad) * math.sin(roll_rad)))
    cv2.arrowedLine(img, origin, z_end, (255, 0, 0), 2)
    
    # Add labels
    cv2.putText(img, f"Roll: {roll:.1f}°", (x - 45, y + size + 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(img, f"Pitch: {pitch:.1f}°", (x - 45, y + size + 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(img, f"Yaw: {yaw:.1f}°", (x - 45, y + size + 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

def main():
    # Create a background image for the dashboard
    dashboard_width = 1280
    dashboard_height = 720
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Create the application window
    cv2.namedWindow("SmartStudy Focus Monitor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SmartStudy Focus Monitor", dashboard_width, dashboard_height)
    
    # Get frame dimensions
    _, frame = cap.read()
    frame_height, frame_width = frame.shape[:2]
    frame_size = (frame_height, frame_width)
    
    # Initialize head pose estimator
    pose_estimator = HeadPoseEstimator()
    
    # Variables for blink detection
    blink_counter = 0
    blink_start_time = time.time()
    blink_rate = 0
    
    # Blink state tracking
    in_blink = False
    potential_blink_start = 0
    last_blink_time = 0
    ear_min_during_blink = 1.0
    
    # Baseline EAR (when eyes are open)
    baseline_ear = None
    ear_baseline_samples = []
    
    # For tracking EAR and head pose history
    ear_history = []
    head_movement_history = []
    max_history = 150  # Last 150 frames
    
    # For focus score history
    focus_scores = []
    max_focus_history = 100
    
    # Calibration mode
    calibration_mode = True
    calibration_start_time = time.time()
    calibration_countdown = 5  # 5 seconds countdown
    
    # Timing variables
    last_time = time.time()
    fps_history = []
    
    # Main loop
    while True:
        # Calculate FPS
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        fps = 1.0 / dt if dt > 0 else 30.0
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)
        
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create dashboard background
        dashboard = np.zeros((dashboard_height, dashboard_width, 3), dtype=np.uint8)
        
        # Draw application header
        cv2.rectangle(dashboard, (0, 0), (dashboard_width, 60), (40, 40, 40), -1)
        cv2.putText(dashboard, "SmartStudy Focus Monitoring System", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(dashboard, f"FPS: {avg_fps:.1f}", (dashboard_width - 150, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1)
        
        # Create space for camera feed
        camera_x = 20
        camera_y = 80
        camera_width = 640
        camera_height = 480
        
        # Create space for metrics
        metrics_x = camera_x + camera_width + 20
        metrics_y = camera_y
        metrics_width = dashboard_width - metrics_x - 20
        metrics_height = 480
        
        # Draw camera feed background
        cv2.rectangle(dashboard, (camera_x - 2, camera_y - 2), 
                     (camera_x + camera_width + 2, camera_y + camera_height + 2), 
                     (180, 180, 180), 2)
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = detector(gray)
        
        # Default focus score if no face is detected
        focus_score = 0
        
        if len(faces) == 0:
            # No face detected
            # Display message on camera feed
            message = "No face detected"
            cv2.putText(frame, message, (frame_width//2 - 80, frame_height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Add zeros to history
            ear_history.append(0)
            head_movement_history.append(0)
        else:
            # Process the first face found
            face = faces[0]
            landmarks = predictor(gray, face)
            shape = shape_to_np(landmarks)
            
            # Get eye landmarks
            left_eye = shape[36:42]
            right_eye = shape[42:48]
            
            # Calculate EAR for each eye
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            
            # Average EAR between both eyes
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Add to history
            ear_history.append(avg_ear)
            
            # Initialize baseline if needed (using the first 30 frames)
            if baseline_ear is None or len(ear_baseline_samples) < 30:
                ear_baseline_samples.append(avg_ear)
                if len(ear_baseline_samples) >= 30:
                    # Calculate 75th percentile as baseline (when eyes are normally open)
                    sorted_samples = sorted(ear_baseline_samples)
                    baseline_ear = sorted_samples[int(len(sorted_samples) * 0.75)]
                    print(f"Established baseline EAR: {baseline_ear:.3f}")
            
            # Detect blinks based on significant EAR change from baseline
            if baseline_ear is not None:
                # Calculate time since last blink
                time_since_last_blink = current_time - last_blink_time
                
                # Detect start of blink - must drop below threshold
                if not in_blink and time_since_last_blink > COOLDOWN_PERIOD:
                    if avg_ear < (baseline_ear - EAR_CHANGE_THRESHOLD) and avg_ear < 0.18:
                        in_blink = True
                        potential_blink_start = current_time
                        ear_min_during_blink = avg_ear
                
                # Track during blink
                elif in_blink:
                    # Update minimum EAR during blink
                    ear_min_during_blink = min(ear_min_during_blink, avg_ear)
                    
                    # Check if blink has ended (EAR returns close to baseline)
                    if avg_ear > (baseline_ear - EAR_CHANGE_THRESHOLD/2):
                        # Calculate blink duration
                        blink_duration = current_time - potential_blink_start
                        
                        # Validate blink duration
                        if MIN_BLINK_DURATION <= blink_duration <= MAX_BLINK_DURATION:
                            ear_drop = baseline_ear - ear_min_during_blink
                            
                            # Ensure significant drop - must be at least 80% of the threshold
                            if ear_drop > EAR_CHANGE_THRESHOLD * 0.8:
                                blink_counter += 1
                                last_blink_time = current_time
                        
                        # Reset blink state
                        in_blink = False
                    
                    # Abandon if blink is too long
                    elif current_time - potential_blink_start > MAX_BLINK_DURATION:
                        in_blink = False
            
            # Draw face landmarks
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
            # Draw eye contours with larger points for better visibility
            for (x, y) in left_eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            for (x, y) in right_eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                
            # Draw convex hull for eyes
            left_eye_hull = cv2.convexHull(left_eye)
            right_eye_hull = cv2.convexHull(right_eye)
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
            
            # Handle head pose calibration
            if calibration_mode:
                remaining_time = calibration_countdown - (current_time - calibration_start_time)
                
                if remaining_time <= 0:
                    # Get current head pose
                    roll, pitch, yaw = pose_estimator.get_pose(shape, frame_size)
                    
                    if roll is not None:
                        # Perform calibration
                        pose_estimator.calibrate((roll, pitch, yaw))
                        calibration_mode = False
                        
                        # Display confirmation message
                        cv2.putText(frame, "Calibration Complete!", (frame_width//2 - 120, frame_height//2), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    # Show countdown
                    cv2.putText(frame, f"Look straight ahead. Calibrating in {int(remaining_time)}s", 
                               (frame_width//2 - 180, frame_height//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Get head pose
            roll, pitch, yaw = pose_estimator.get_pose(shape, frame_size)
            
            if roll is not None:
                # Calculate total head movement
                total_movement = abs(roll) + abs(pitch) + abs(yaw)
                head_movement_history.append(total_movement)
                
                # Limit history size
                if len(head_movement_history) > max_history:
                    head_movement_history = head_movement_history[-max_history:]
                
                # Calculate blink rate (blinks per minute)
                elapsed_time = current_time - blink_start_time
                if elapsed_time >= 60:
                    blink_rate = blink_counter / (elapsed_time / 60)
                    # Reset for next window
                    blink_counter = 0
                    blink_start_time = current_time
                else:
                    # Estimate based on current data
                    blink_rate = blink_counter / (elapsed_time / 60) if elapsed_time > 0 else 0
                
                # Calculate focus score
                avg_head_movement = sum(head_movement_history) / len(head_movement_history) if head_movement_history else 0
                focus_score = compute_focus_score(avg_ear, blink_rate, avg_head_movement, baseline_ear)
                
                # Add to focus history
                focus_scores.append(focus_score)
                if len(focus_scores) > max_focus_history:
                    focus_scores.pop(0)
            else:
                head_movement_history.append(0)
        
        # Place frame on dashboard
        frame_resized = cv2.resize(frame, (camera_width, camera_height))
        dashboard[camera_y:camera_y+camera_height, camera_x:camera_x+camera_width] = frame_resized
        
        # Draw focus gauge
        gauge_x = metrics_x
        gauge_y = metrics_y
        gauge_width = metrics_width
        gauge_height = 50
        draw_focus_gauge(dashboard, focus_score, gauge_x, gauge_y, gauge_width, gauge_height)
        
        # Draw focus status
        focus_status, focus_color = get_focus_status(focus_score)
        cv2.putText(dashboard, "Focus Score:", (gauge_x, gauge_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
        cv2.putText(dashboard, focus_status, (gauge_x + gauge_width//2 - len(focus_status)*5, gauge_y + gauge_height + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, focus_color, 2)
        
        # Draw metric boxes
        box_width = metrics_width
        box_height = 100
        margin = 20
        
        # EAR metric box
        ear_value = avg_ear if 'avg_ear' in locals() else 0
        ear_status, ear_color = evaluate_ear(ear_value)
        draw_metric_box(dashboard, "Eye Openness (EAR)", f"{ear_value:.2f}", 
                       ear_status, ear_color, metrics_x, metrics_y + 80, box_width, box_height)
        
        # Blink rate metric box
        blink_rate_value = blink_rate if 'blink_rate' in locals() else 0
        if BLINK_OPTIMAL_RANGE[0] <= blink_rate_value <= BLINK_OPTIMAL_RANGE[1]:
            blink_status = "Optimal"
            blink_color = (0, 180, 0)
        elif blink_rate_value < BLINK_OPTIMAL_RANGE[0]:
            blink_status = "Low Rate"
            blink_color = (180, 180, 0)
        else:
            blink_status = "High Rate"
            blink_color = (0, 0, 180)
        draw_metric_box(dashboard, "Blink Rate", f"{blink_rate_value:.1f}/min", 
                       blink_status, blink_color, metrics_x, metrics_y + 80 + box_height + margin, box_width, box_height)
        
        # Head movement metric box
        if 'roll' in locals() and roll is not None:
            head_value = abs(roll) + abs(pitch) + abs(yaw)
            if head_value < HEAD_MOVEMENT_THRESHOLD / 2:
                head_status = "Stable"
                head_color = (0, 180, 0)
            elif head_value < HEAD_MOVEMENT_THRESHOLD:
                head_status = "Moderate"
                head_color = (180, 180, 0)
            else:
                head_status = "Excessive"
                head_color = (0, 0, 180)
            head_text = f"{head_value:.1f}°"
        else:
            head_status = "Not Detected"
            head_color = (180, 0, 0)
            head_text = "N/A"
        draw_metric_box(dashboard, "Head Movement", head_text, 
                       head_status, head_color, metrics_x, metrics_y + 80 + 2*(box_height + margin), box_width, box_height)
        
        # Draw the head pose visualization if face is detected
        if 'roll' in locals() and roll is not None:
            head_viz_x = metrics_x + box_width//2
            head_viz_y = metrics_y + 80 + 3*(box_height + margin) + 40
            draw_head_pose_visualization(dashboard, roll, pitch, yaw, head_viz_x, head_viz_y, 60)
        
        # Draw focus history graph
        if focus_scores:
            graph_x = camera_x
            graph_y = camera_y + camera_height + 20
            graph_width = dashboard_width - 40
            graph_height = 120
            
            # Draw graph background
            cv2.rectangle(dashboard, (graph_x, graph_y), 
                         (graph_x + graph_width, graph_y + graph_height), 
                         (60, 60, 60), -1)
            cv2.rectangle(dashboard, (graph_x, graph_y), 
                         (graph_x + graph_width, graph_y + graph_height), 
                         (180, 180, 180), 2)
            
            # Draw graph title
            cv2.putText(dashboard, "Focus History", (graph_x + 10, graph_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
            
            # Draw Y-axis labels
            for i in range(0, 101, 20):
                y_pos = graph_y + graph_height - int((i / 100) * (graph_height - 20)) - 10
                cv2.putText(dashboard, str(i), (graph_x - 25, y_pos + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
                cv2.line(dashboard, (graph_x, y_pos), (graph_x + graph_width, y_pos), 
                        (100, 100, 100), 1)
            
            # Draw focus score line
            points = []
            for i, score in enumerate(focus_scores):
                x = graph_x + 10 + int(i * (graph_width - 20) / max_focus_history)
                y = graph_y + graph_height - int((score / 100) * (graph_height - 20)) - 10
                points.append((x, y))
            
            # Draw the line
            if len(points) > 1:
                for i in range(1, len(points)):
                    # Color based on score value
                    score = focus_scores[i]
                    _, color = get_focus_status(score)
                    cv2.line(dashboard, points[i-1], points[i], color, 2, cv2.LINE_AA)
            
            # Add timestamp indicators
            cv2.putText(dashboard, "Now", (graph_x + graph_width - 30, graph_y + graph_height - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
            cv2.putText(dashboard, f"-{len(focus_scores)/30:.1f}min", (graph_x + 5, graph_y + graph_height - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
        
        # Draw version and company information
        cv2.putText(dashboard, "SmartStudy v1.0", (20, dashboard_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(dashboard, "© 2025 Smart Study Technologies", (dashboard_width - 300, dashboard_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Display dashboard
        cv2.imshow("SmartStudy Focus Monitor", dashboard)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()