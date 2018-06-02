from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras import models
from tensorflow.python.keras import layers

import numpy as np


class BinaryClassifier:

    def __init__(self, model: models.Sequential=None):
        if model is None:
            model = models.Sequential()

        self.model = model

    def train_model(self, data: list, n_words: int):
        features, labels = self.__process_text(data, n_words)
        self.model.add(layers.Dense(16, activation='relu', input_shape=(n_words,)))
        self.model.add(layers.Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(features, labels, epochs=10, verbose=2)

    def evaluate_model(self, data: list, n_words: int):
        features, labels = self.__process_text(data, n_words)
        _, acc = self.model.evaluate(features, labels, verbose=0)
        return acc * 100

    @staticmethod
    def __process_text(items: list, num_words: int):
        raw_features = []
        raw_labels = []
        negative = 0
        positive = 1

        # Split Features and Labels
        for item in items:
            raw_features.append(item[0])
            raw_labels.append(positive if item[1] is True else negative)

        # Vectorise Features
        tokenizer = Tokenizer(num_words=num_words)
        tokenizer.fit_on_texts(raw_features)
        features = tokenizer.texts_to_matrix(raw_features, mode='binary')  # one hot results

        # Vectorise Labels
        labels = np.asarray(raw_labels).astype('float32')

        return features, labels
