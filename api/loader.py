import numpy as np

# классы, на которые обучена LSTM модель
actions = np.array([
    'a',
    'b',
    'c',
    'd',
    'e',
    'f',
    'g',
    'h',
    'i',
    'j',
    'NaN',
    'k',
    'l',
    'm',
    'n',
    'o',
    'p',
    'q',
    'r',
    's',
    't',
    'u',
    'v',
    'w',
    'x',
    'y',
    'z',
    'like',
    'I(me)',
    'thank you',
    'eat',
    'play',
    'clothes',
    'boots',
    'motorcycle',
    'your',
    'need',
    'model_start',
    'model_stop',
])


# Цвета для визуализации
colors = []
for i in range(len(actions)):
    colors.append(tuple(map(int, np.random.choice(range(256), size=3))))

# Путь для основной папки, где будет храниться датасет
path = 'Data1'

# Количество видео по каждому классу(знаку) для обучения
no_sequences = 90

# Длина(кол-во кадров) каждого видео
sequence_length = 30
