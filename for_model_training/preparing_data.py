import os
import numpy as np
from tensorflow.keras.utils import to_categorical

from api.loader import path, actions, sequence_length


def preparing_data(data_path=path, signs_names=actions, seq_length=sequence_length):
    """
    Функция для подготовки данных к работе с LSTM моделью
    :param data_path: путь для основной папки
    :param signs_names: классы(знаки), на которые будет обучена модель
    :param seq_length: Длина(кол-во кадров) каждого видео
    :return: данные для обучения модели: вектора, состоящие из координат точек
    """
    label_map = {label: num for num, label in enumerate(signs_names)}

    sequences, labels = [], []
    for action in signs_names:
        file_number = len([i for i in os.listdir(f'{data_path}/{action}')])
        for sequence in range(file_number):
            window = []
            sequence = str(sequence)
            for frame_num in range(seq_length):
                res = np.load(f'{data_path}/{action}/{sequence}/{frame_num}.npy')
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    return X, y
