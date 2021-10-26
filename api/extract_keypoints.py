import numpy as np


def extract_keypoints(results):
    """
    Функция создает вектор-строку из координат ключевых точек обеих рук с одного кадра изображения. У каждой руки 21 точка
    :param results: объект mediapipe, который был получен при обработке их моделью кадра изображения.
    Получен в результате отработки функции 'mediapipe_detection'
    :return: вектор из координат точек
    """
    if results.right_hand_landmarks:  # если правая рука была задетектирована, то существует такой объект
        # центрируем первую точку к координатам (0.5, 0.5) и переносим все остальные точки относительно нее
        rh_centr_x = 0.5 - results.right_hand_landmarks.landmark[0].x
        rh_centr_y = 0.5 - results.right_hand_landmarks.landmark[0].y

        rh = np.array(
            [[res.x + rh_centr_x, res.y + rh_centr_y] for res in results.right_hand_landmarks.landmark]).flatten()
    else:  # если правая рука не была задетектирована, то создаем вектор из координат-нулей
        rh = np.zeros(21 * 2)

    if results.left_hand_landmarks:  # если левая рука была задетектирована, то существует такой объект
        # центрируем первую точку к координатам (0.5, 0.5) и переносим все остальные точки относительно нее
        lh_centr_x = 0.5 - results.left_hand_landmarks.landmark[0].x
        lh_centr_y = 0.5 - results.left_hand_landmarks.landmark[0].y

        lh = np.array(
            [[res.x + lh_centr_x, res.y + lh_centr_y] for res in results.left_hand_landmarks.landmark]).flatten()
    else:  # если левая рука не была задетектирована, то создаем вектор из координат-нулей
        lh = np.zeros(21 * 2)
    return np.concatenate([rh, lh])  # объединяем координаты точек обеих рук в один вектор-строку
