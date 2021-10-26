import os
import cv2
import numpy as np

from api.loader import path, no_sequences, sequence_length
from api.mediapipe_model import mp_holistic, mediapipe_detection, draw_styled_landmarks
from api.extract_keypoints import extract_keypoints


def create_data_for_one_sign(sign_name, data_path=path, number_of_sequences=no_sequences, seq_length=sequence_length):
    """
    Функция для записи данных для одного класса(знака)
    :param sign_name: название класса
    :param data_path: путь для основной папки
    :param number_of_sequences: количество видео, которые будут записанны под класс
    :param seq_length: длина(кол-во кадров) одного видео
    :return:
    """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    cv2.namedWindow(f"recording data for '{sign_name}' sign")
    cv2.moveWindow(f"recording data for '{sign_name}' sign", 0, 0)
    with mp_holistic.Holistic(model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for sequence in range(number_of_sequences):
            for frame_num in range(seq_length):

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks
                draw_styled_landmarks(image, results)

                # Apply wait logic
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Collecting frames for "{sign_name}" Video Number {sequence}', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow(f"recording data for '{sign_name}' sign", image)
                    # pause before next video recording
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, f'Collecting frames for "{sign_name}" Video Number {sequence}', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow(f"recording data for '{sign_name}' sign", image)

                # Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(data_path, sign_name, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                if cv2.getWindowProperty(f"recording data for '{sign_name}' sign", cv2.WND_PROP_VISIBLE) < 1:
                    break

        cap.release()
        cv2.destroyAllWindows()


create_data_for_one_sign('a')
