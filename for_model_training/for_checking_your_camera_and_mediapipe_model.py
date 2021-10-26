# Запустите этот скрипт, если хотите проверить, как отрабатывает mediapipe модель

import cv2

from api.mediapipe_model import mp_holistic, mediapipe_detection, draw_styled_landmarks

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
cv2.namedWindow("camera check")
cv2.moveWindow("camera check", 0, 0)

with mp_holistic.Holistic(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)

        # Draw landmarks
        draw_styled_landmarks(image, results)

        # Show to screen
        cv2.imshow('camera check', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty("camera check", cv2.WND_PROP_VISIBLE) < 1:
            break
    cap.release()
    cv2.destroyAllWindows()
