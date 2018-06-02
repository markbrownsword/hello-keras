from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras import models
from tensorflow.python.keras import layers

import numpy as np


class BinaryClassifier:

    def __init__(self, model: models.Sequential=None, tokenizer: Tokenizer=None):
        if model is None:
            model = models.Sequential()

        if tokenizer is None:
            tokenizer = Tokenizer()

        self.model = model
        self.tokenizer = tokenizer

    def train_model(self, data: list):
        features, labels = self.__process_text(data, True)
        n_words = features.shape[1]

        self.model.add(layers.Dense(16, activation='relu', input_shape=(n_words,)))
        self.model.add(layers.Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(features, labels, epochs=10, verbose=2)

    def evaluate_model(self, data: list):
        features, labels = self.__process_text(data)
        _, acc = self.model.evaluate(features, labels, verbose=0)
        return acc * 100

    def __process_text(self, items: list, fit_tokenizer: bool=False):
        raw_features = []
        raw_labels = []
        negative = 0
        positive = 1

        # Split Features and Labels
        for item in items:
            raw_features.append(item[0])
            raw_labels.append(positive if item[1] is True else negative)

        # Fit tokenizer
        if fit_tokenizer:
            self.tokenizer.fit_on_texts(raw_features)

        # Vectorise Features
        features = self.tokenizer.texts_to_matrix(raw_features, mode='binary')  # one hot results

        # Vectorise Labels
        labels = np.asarray(raw_labels).astype('float32')

        return features, labels
