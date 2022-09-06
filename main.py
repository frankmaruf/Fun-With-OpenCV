
import cv2, mediapipe as mp

vid = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh()
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
while(True):
    ret, frame = vid.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    edged = cv2.Canny(gray, 30, 200)
    output = face_mesh.process(rgb_frame)
    landMark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape
    # Hands
    hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    frame.flags.writeable = False
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
          for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    if landMark_points:
        landMarks = landMark_points[0].landmark
        for landMark in landMarks:
            x = int(landMark.x * frame_w)
            y = int(landMark.y * frame_h)
            cv2.circle(frame,(x,y),3,(0,255,0))
            cv2.putText(frame,"Human",(00,185),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(100,0,255),thickness=1)
    contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()