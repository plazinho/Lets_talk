import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

from for_model_training.preparing_data import preparing_data
from api.loader import actions, sequence_length


def train_lstm_model(signs_names=actions, epochs=300, batch_size=512, seq_length=sequence_length):
    """
    Функция для обучения LSTM модели и сохранения ее весов
    :param signs_names: классы(знаки), на которые будет обучена модель
    :param epochs: количество эпох обучения
    :param batch_size: размер батча
    :param seq_length: Длина(кол-во кадров) каждого видео
    :return:
    """
    X, y = preparing_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    model = Sequential()
    # 30 кадров(изображений) и 84 координаты: 2 руки * 21 точка * 2 координаты(x и y)
    model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(seq_length, 84)))
    model.add(LSTM(256, return_sequences=True, activation='relu'))
    model.add(LSTM(128, return_sequences=False, activation='relu'))

    model.add(Dense(signs_names.shape[0], activation='softmax'))

    optim = Adam(clipnorm=0.7)

    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    yhat = model.predict(X_test)
    ytrue = actions[np.argmax(y_test, axis=1)]
    yhat = actions[np.argmax(yhat, axis=1)]

    print([(i, j) for i, j in zip(ytrue, yhat) if i != j])
    model.save(f'3lstm_{epochs}epochs.h5')


train_lstm_model()
