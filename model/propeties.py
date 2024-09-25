import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.datasets import imdb # type: ignore
from sklearn.model_selection import train_test_split 

def init_model():
    # Загрузка набора данных
    # Паддинг последовательностей до одинаковой длины
    max_length = 500 
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000)
    x_train = pad_sequences(x_train, maxlen=max_length)
    x_test = pad_sequences(x_test, maxlen=max_length)

    # Создание модели типа LSTM
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=128, input_length=max_length))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))

    # Коомпиляция модели
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Обучение модели
    history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test), verbose=2)

    # Оценка модели на тестовых данных
    score, accuracy = model.evaluate(x_test, y_test, batch_size=64)
    print('Точность модели: {:.2f}%'.format(accuracy*100))

    def predict(text):
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts([text])
        sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=max_length)
        prediction = model.predict(padded_sequence)
        return "Positive" if prediction[0][0] > 0.5 else "Negative"
    
    def input_text():
        text = input("Print your text: ")
        print(predict(text))