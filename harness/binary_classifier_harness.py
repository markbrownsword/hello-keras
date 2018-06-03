from hellokeras.binary_classifier import BinaryClassifier

training_corpus = [
    ('I am exhausted of this work.', False),
    ("I can't cooperate with this", False),
    ('He is my badest enemy!', False),
    ('My management is poor.', False),
    ('I love this burger.', True),
    ('This is an brilliant place!', True),
    ('I feel very good about these dates.', True),
    ('This is my best work.', True),
    ("What an awesome view", True),
    ('I do not like this dish', False)
]

test_corpus = [
    ("I am not feeling well today.", False),
    ("I feel brilliant!", True),
    ('Gary is a friend of mine.', True),
    ("I can't believe I'm doing this.", False),
    ('The date was good.', True),
    ('I do not enjoy my job', False)
]

# Initialise Binary Classifier
binary_classifier = BinaryClassifier()

# Train Model
binary_classifier.train(training_corpus)

# Evaluate Model
accuracy = binary_classifier.evaluate(test_corpus)
print('Test Accuracy: %f ' % (accuracy * 100))

# Predict
percent, sentiment = binary_classifier.predict('It\'s getting harder')
print('Sentiment: %s (%.2f%%) ' % (sentiment, percent * 100))
