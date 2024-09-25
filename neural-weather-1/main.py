import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Шаг 1: Подготовка данных
# Пример случайных данных: температура (°C), влажность (%), давление (гПа), скорость ветра (м/с)
X_train = np.array([[22, 70, 1012, 3],
                    [25, 60, 1010, 4],
                    [18, 85, 1015, 2],
                    [30, 50, 1008, 5]])

# Выходные данные: 1 - дождь, 0 - без дождя
y_train = np.array([1, 0, 1, 0])

# Шаг 2: Построение нейронной сети
model = Sequential()

# Входной слой с 4 нейронами (количество фичей) и скрытый слой с 8 нейронами
model.add(Dense(8, input_dim=4, activation='relu'))

# Ещё один скрытый слой
model.add(Dense(8, activation='relu'))

# Выходной слой с 1 нейроном (будет предсказывать вероятность дождя)
model.add(Dense(1, activation='sigmoid'))

# Шаг 3: Компиляция модели
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Шаг 4: Обучение модели
model.fit(X_train, y_train, epochs=100, verbose=1)

# Шаг 5: Прогноз на новые данные
X_new = np.array([[18, 64, 1010, 6]])  # Новые данные для предсказания
prediction = model.predict(X_new)

print(f"Вероятность дождя: {prediction[0][0]:.2f}")
