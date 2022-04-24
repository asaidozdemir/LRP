import random
import cv2 as cv
import keras.datasets.mnist as mnist
import matplotlib.pyplot as plt
import numpy as np
from sklearn import ensemble
from sklearn.datasets import load_digits

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

####################################################################

(x_train, y_train), (x_test, y_test) = mnist.load_data()

##################################################
train_target_z = y_train.tolist()
flat_image = []
flat_array = []
for i in range(0, 60000):
    flat_image = np.ravel(x_train[i])
    flat_image = np.asarray(flat_image)
    flat_array.append(flat_image)
train_images_z = np.asarray(flat_array)
del flat_image
del flat_array
##################################################
test_target_z = y_test.tolist()
flat_image = []
flat_array = []
for i in range(0, 10000):
    flat_image = np.ravel(x_test[i])
    flat_image = np.asarray(flat_image)
    flat_array.append(flat_image)
test_images_z = np.asarray(flat_array)
del flat_image
del flat_array
##################################################

del test_index
del train_index
del sklearn_length
del digits
del sklearn_x
del sklearn_y

del x_train
del y_train
del x_test
del y_test

####################################################################


# Using the Random Forest Classifier
classifier = ensemble.RandomForestClassifier()

# Fit model with sample data
classifier.fit(train_images_z, train_target_z)

# Attempt to predict validation data
score = classifier.score(test_images_z, test_target_z)
print('Random Tree Classifier:\n')
print('Score\t' + str(score))

for qqq in range(0, 10):
    img = cv.imread(f'28by28/{qqq}.png')[:, :, 0]
    img = np.invert(np.array([img]))
    prediction = classifier.predict(img.reshape(1, -1))
    print(f'The result is probably: {np.max(prediction)}')
    print(f'The Image Was: {qqq}\n')
    plt.imshow(img[0], cmap=plt.get_cmap('gray'))
    plt.show()
