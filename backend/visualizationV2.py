import cv2
import mediapipe as mp
import numpy as np


## NEED TO LEARN THIS 

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate angle at point b
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Pose detection loop
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB for MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Pose detection
        results = pose.process(image)

        # Convert back to BGR for OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            # Draw skeleton
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245,255,255), thickness=2, circle_radius=2)
            )

            # Extract landmarks
            lm = results.pose_landmarks.landmark
            h, w, _ = image.shape

            # Key points
            nose = [lm[0].x * w, lm[0].y * h]
            left_shoulder = [lm[11].x * w, lm[11].y * h]
            right_shoulder = [lm[12].x * w, lm[12].y * h]

            # Calculate angles
            left_shoulder_angle = calculate_angle(nose, left_shoulder, right_shoulder)
            right_shoulder_angle = calculate_angle(nose, right_shoulder, left_shoulder)
            nose_angle = calculate_angle(left_shoulder, nose, right_shoulder)

            # Display angles on video
            cv2.putText(image, f"{int(left_shoulder_angle)}", tuple(np.int32(left_shoulder)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
            cv2.putText(image, f"{int(right_shoulder_angle)}", tuple(np.int32(right_shoulder)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
            cv2.putText(image, f"{int(nose_angle)}", tuple(np.int32(nose)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)

        # Show the video
        cv2.imshow('Posture Cam', image)

        # Quit on 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
