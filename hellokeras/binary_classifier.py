from tensorflow.python.keras.preprocessing import text
from tensorflow.python.keras import models
from tensorflow.python.keras import layers

import numpy as np


class BinaryClassifier:

    def __init__(self):
        self.model = models.Sequential()
        self.tokenizer = text.Tokenizer()

    def train(self, data: list):
        features, labels = self.__process_text(data, True)
        n_words = features.shape[1]
        self.model.add(layers.Dense(16, activation='relu', input_shape=(n_words,)))
        self.model.add(layers.Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(features, labels, epochs=10, verbose=2)

    def evaluate(self, data: list):
        features, labels = self.__process_text(data)
        _, accuracy = self.model.evaluate(features, labels, verbose=0)
        return accuracy

    def predict(self, input_text: str):
        raw_features = [input_text]
        features = self.__vectorize_features(raw_features)
        predictions = self.model.predict(features)
        result = predictions[0, 0]
        if round(result) == 0:
            return result, 'NEGATIVE'
        return result, 'POSITIVE'

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
        features = self.__vectorize_features(raw_features)

        # Vectorise Labels
        labels = np.asarray(raw_labels).astype('float32')

        return features, labels

    def __vectorize_features(self, features: list):
        return self.tokenizer.texts_to_matrix(features, mode='binary')  # one hot results
