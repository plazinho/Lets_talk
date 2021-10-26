import numpy as np
import cv2
from PyQt5 import QtWidgets

from api.mediapipe_model import mp_holistic, mediapipe_detection, draw_styled_landmarks
from api.extract_keypoints import extract_keypoints
from api.LSTM_model import model
from api.loader import actions, colors
from api.vizualization import prob_viz

from ui_editor.start_ui import Ui_Start

# Список, в который будут записываться предсказания модели - буквы и слова. Также это "строка предсказаний" в верхней части экрана видеокамеры
sentence = []

# Список, в который будут записываться только предсказания по классам "model_start" и "model_stop",
# чтобы в нужный момент времени иметь возможность определить, в каком режиме работает модель
start_pause_predictions = []


def camera_start():
    """
    Главная функция, внутри которой прописана логика предсказания LSTM модели
    :return:
    """
    # Обозначаем глобальные переменные, чтобы иметь возможность всегда к ним обращаться
    global sentence, start_pause_predictions

    # Список, в который будут записываться все предсказания модели в виде слов, предложение за предложением.
    # Также это текст в "окне предсказаний" в окне, которое будет открываться справа от главного окна видеокамеры
    text = []

    # Список, в котором будут храниться координаты точек рук по каждому кадру
    sequence = []

    # Список, в котором будут храниться результаты предсказаний модели, а точнее индексы максимального предсказания(argmax)
    # из списка 'res' - списка вероятностей предсказаний LSTM модели
    predictions = []

    # Если предсказание модели превысит эту границу 'threshold', то оно будет записано в список 'predictions'
    threshold = 0.7

    # Создаем объект видеокамеры и устанавливаем размер окна
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

    # Именуем окно с видеокамерой и передвигаем его левый верхний угол в указанные координаты
    cv2.namedWindow("Let's talk")
    cv2.moveWindow("Let's talk", 0, 0)

    # Инициализация окна для записи всех предложений переменной 'text' в 'окно предсказаний' справа от главного окна с видеокамерой
    Form = QtWidgets.QWidget()
    ui_start = Ui_Start()
    ui_start.setupUi(Form)
    Form.show()

    with mp_holistic.Holistic(model_complexity=1, min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Считываем один кадр
            ret, frame = cap.read()

            # Отработка предобученной модели mediapipe. Запись объектов, содержащие координаты ключевых точек, в переменную 'results'
            image, results = mediapipe_detection(frame, holistic)

            # Визуализация ключевых точек на руках
            draw_styled_landmarks(image, results)

            # Получаем координаты точек с кадра
            keypoints = extract_keypoints(results)

            # Записываем их в список "sequence"
            sequence.append(keypoints)

            # Берем координаты точек только с последних 30-и кадров, т.к. такова размерность наших входных данных для LSTM модели
            sequence = sequence[-30:]

            # Отрисовываем на экране зеленый круг, если модель находиться в режиме 'model_start'
            if len(start_pause_predictions) == 0 or start_pause_predictions[-1] == 'model_start':
                cv2.circle(image, (1190, 90), 24, (0, 0, 255), -1)
            # В противном случае отрисовываем красный круг
            else:
                cv2.circle(image, (1190, 90), 24, (0, 255, 0), -1)
            if len(sequence) == 30:  # если накопилось 30 кадров
                res = model.predict(np.expand_dims(sequence, axis=0))[0]  # модель делает предсказания по этим 30-и кадрам
                current_prediction = np.argmax(res)  # индекс максимального предсказания
                name_current_prediction = actions[current_prediction]  # класс, который предсказала модель

                if res[current_prediction] > threshold:  # если уверенность данного предсказания превосходит границу
                    predictions.append(current_prediction)  # добавляем это предсказание
                    last_predictions = np.unique(predictions[-12:])  # смотрим, сколько уникальных предсказаний было из последних 12-и предсказаний

                    # если все 12 последних предсказаний были одинаковы и они совпадают с текущим предсказанием
                    if len(last_predictions) == 1 and last_predictions[0] == current_prediction:
                        # если класс текущего предсказания был 'model_start' или 'model_stop', то записываем это в отдельный список
                        if name_current_prediction == 'model_start' or name_current_prediction == 'model_stop':
                            start_pause_predictions.append(name_current_prediction)
                        # если модель находится в режиме 'model_start'
                        if len(start_pause_predictions) == 0 or start_pause_predictions[-1] == 'model_start':
                            # если класс текущего предсказания 'NaN' (нет рук в кадре)
                            if name_current_prediction == 'NaN':
                                # проверка, есть ли уже записанные слова или буквы
                                if len(sentence) > 0:
                                    # если нет, то записываем все предсказания(слова или буквы из переменной 'sentence')
                                    # в переменную 'text'
                                    fixed = [i if len(i) == 1 else f' {i} ' for i in sentence]
                                    text.extend(''.join(fixed).strip().replace('  ', ' ').split())
                                    text[-1] = text[-1] + '.'
                                    ui_start.start_window.setPlainText(' '.join(text))  # обновляем 'окно записи'
                                    sentence = [] # обнуляем переменную, т.к. предложение были уже записано в 'text'

                            # в противном случае если только класс текущего предсказания не 'model_start'
                            elif name_current_prediction != 'model_start':
                                # если уже есть записанные слова или буквы
                                if len(sentence) > 0:
                                    # если класс текущего предсказания не равен классу последнему записанному предсказанию
                                    # (чтобы избежать повторной записи одних и тех же подряд предсказанных классов)
                                    if name_current_prediction.lower() != sentence[-1].lower():
                                        # то записываем класс текущего предсказания
                                        sentence.append(name_current_prediction)
                                # если записанных слов или букв еще нет(то есть сейчас будет производится первая запись)
                                else:
                                    # то также записываем класс текущего предсказания, но с большой буквы
                                    sentence.append(name_current_prediction.capitalize())

                # Визуализируем предсказания
                image = prob_viz(res, actions, image, colors)
            # Отрисовываем "строку предсказаний" вверху главного окна видеокамеры
            cv2.rectangle(image, (0, 0), (1280, 40), (188, 143, 143), -1)
            # В "строке предсказаний" будем показывать только 15 последних записей
            cv2.putText(image, ' '.join(sentence[-15:]), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Показываем итоговое изображение со всей отрисовкой и визаулизацией
            cv2.imshow("Let's talk", image)
            # если просходит нажатие кнопки 'q', то цикл предсказаний прерывается
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            # или по нажатию кнопки закрытия окна
            if cv2.getWindowProperty("Let's talk", cv2.WND_PROP_VISIBLE) < 1:
                break

        # освобождаем камеру и закрываем главное окно с видеокамерой
        cap.release()
        cv2.destroyAllWindows()
        # закрываем окно справа от главного окна с видеокамерой
        Form.close()
