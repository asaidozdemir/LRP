import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import random
from sklearn import ensemble
import numpy as np
import cv2 as cv

digits = load_digits()

# Define variables
sklearn_length = len(digits.images)
sklearn_x = digits.images.reshape((sklearn_length, -1))
sklearn_y = digits.target

# Create random indices
test_index = random.sample(range(len(sklearn_x)), len(sklearn_x) // 5)  # 20-80
train_index = [i for i in range(len(sklearn_x)) if i not in test_index]

# Sample and validation images
test_images = [sklearn_x[i] for i in test_index]
train_images = [sklearn_x[i] for i in train_index]

# Sample and validation targets
test_target = [sklearn_y[i] for i in test_index]
train_target = [sklearn_y[i] for i in train_index]

# Using the Random Forest Classifier
classifier = ensemble.RandomForestClassifier()

# Fit model with sample data
classifier.fit(train_images, train_target)

# Attempt to predict validation data
score = classifier.score(test_images, test_target)
print('Random Tree Classifier:\n')
print('Score\t' + str(score))

# for qqq in range(0, 1):
#     img = digits.images[qqq]
#     prediction = classifier.predict(img.reshape(1, -1))
#     print(f'The result is probably: {np.max(prediction)}')
#     print(f'The Image Was: {digits.target[qqq]}')
#     plt.imshow(img, cmap=plt.get_cmap('gray'))
#     plt.show()

for qqq in range(0, 10):
    img = cv.imread(f'8by8/{qqq}.png')[:, :, 0]
    img = np.invert(np.array([img]))
    prediction = classifier.predict(img.reshape(1, -1))
    print(f'The result is probably: {np.max(prediction)}')
    print(f'The Image Was: {qqq}\n')
    plt.imshow(img[0], cmap=plt.get_cmap('gray'))
    plt.show()