import tensorflow as tf
from tensorflow.keras import layers, models

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Преобразование изображений 28x28 в плоский вектор
    layers.Dense(128, activation='relu'),  # Полносвязный слой с 128 нейронами и активацией ReLU
    layers.Dense(10, activation='softmax') # Выходной слой с 10 нейронами для каждой цифры (0-9)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
