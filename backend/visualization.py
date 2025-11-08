import cv2
import mediapipe as mp
import numpy as np ## array, helps us with trig
mp_drawing = mp.solutions.drawing_utils ##Visualizing our poses with this 
mp_pose = mp.solutions.pose ## This is importing our pose estimation model.


## Video Feed, open webcam, read through its frames, and display it
cap = cv2.VideoCapture(0) ##Setting up our video capture device (Webcam)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose: ##The sensitivity of your pose
  while cap.isOpened():
    ret, frame = cap.read()
    #cv2.imshow('Posture Cam', frame) ##This is the popup
    
    # Detect stuff and render
    # RECOLOR IMAGE TO RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Make detection, makes use of pose from top var
    results = pose.process(image)

    # RECOLOR IMAGE TO BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # EXTRACT LANDMARKS
    # If landmark is not visible then pass,
    try:
      landmarks = results.pose_landmarks.landmark
    except:
      pass

    # Render detections, pre defined in mediapipe
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(245,117,66), thickness = 2, circle_radius = 4),
                              mp_drawing.DrawingSpec(color=(245,255,255), thickness = 4, circle_radius = 2)
                              )

    # Shows the IMAGE
    cv2.imshow('Posture Cam', image)

    ## Clear the feed on conditions
    if cv2.waitKey(10) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()


  