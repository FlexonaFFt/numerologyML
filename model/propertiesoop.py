import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.datasets import imdb # type: ignore

class SentimentAnalyzer:
    def __init__(self, max_length=500, num_words=5000):
        self.max_length = max_length
        self.num_words = num_words
        self.model = None
        self.tokenizer = Tokenizer(num_words=self.num_words)

    def load_data(self):
        # Загрузка набора данных IMDb
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=self.num_words)
        x_train = pad_sequences(x_train, maxlen=self.max_length)
        x_test = pad_sequences(x_test, maxlen=self.max_length)
        return (x_train, y_train), (x_test, y_test)

    def build_model(self):
        # Создание модели типа LSTM
        self.model = Sequential()
        self.model.add(Embedding(input_dim=self.num_words, output_dim=128))  # Удален input_length
        self.model.add(SpatialDropout1D(0.2))
        self.model.add(LSTM(100))
        self.model.add(Dense(1, activation='sigmoid'))

        # Компиляция модели
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self):
        (x_train, y_train), (x_test, y_test) = self.load_data()
        
        # Обучение модели
        history = self.model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test), verbose=2)

        # Оценка модели на тестовых данных
        score, accuracy = self.model.evaluate(x_test, y_test, batch_size=64)
        print('Точность модели: {:.2f}%'.format(accuracy * 100))

    def predict(self, text):
        # Токенизация и паддинг нового текста
        self.tokenizer.fit_on_texts([text])
        sequence = self.tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_length)

        # Предсказание тональности
        prediction = self.model.predict(padded_sequence)
        return "Positive" if prediction[0][0] > 0.5 else "Negative"

    def input_text(self):
        text = input("Введите ваш текст: ")
        print(self.predict(text))

def init_model():
    analyzer = SentimentAnalyzer()
    analyzer.build_model()
    analyzer.train()