from hellokeras.binary_classification import BinaryClassification


binary_classification = BinaryClassification()
result = binary_classification.execute('It\'s getting harder')

print('result :: {}'.format(result))