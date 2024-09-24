#type: ignore
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Загрузка данных MNIST
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Преобразование данных в плоский формат и нормализация
X_train = X_train.reshape(-1, 28 * 28).astype('float32') / 255
X_test = X_test.reshape(-1, 28 * 28).astype('float32') / 255

# Обучение модели
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Прогнозирование на тестовых данных
y_pred = model.predict(X_test_scaled)

# Оценка точности
accuracy = accuracy_score(y_test, y_pred)
print(f'Точность модели: {accuracy * 100:.2f}%')

# Визуализация первых 10 изображений из тестовой выборки и их предсказаний
plt.figure(figsize=(10, 4))
for index in range(10):
    plt.subplot(2, 5, index + 1)
    plt.imshow(X_test.iloc[index].values.reshape(28, 28), cmap='gray')
    plt.title(f'Предсказано: {y_pred[index]}')
    plt.axis('off')
plt.show()
