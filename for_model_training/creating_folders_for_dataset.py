import os

from api.loader import path, actions, no_sequences


def create_folders(data_path=path, signs_names=actions, number_of_sequences=no_sequences):
    """
    Функция для создания папок под датасет для обучения LSTM модели
    :param data_path: путь для основной папки
    :param signs_names: классы(знаки), на которые будет обучена модель
    :param number_of_sequences: количество видео, которые будут записанны под каждый класс
    :return:
    """
    for name in signs_names:
        for sequence in range(number_of_sequences):
            try:
                os.makedirs(os.path.join(data_path, name, str(sequence)))
            except:
                pass

create_folders()
