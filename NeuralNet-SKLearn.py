import random
import cv2 as cv
import keras.datasets.mnist as mnist
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits

digits = load_digits()

# Define variables
sklearn_length = len(digits.images)
sklearn_x = digits.images
sklearn_y = digits.target

# Create random indices
test_index = random.sample(range(len(sklearn_x)), len(sklearn_x) // 5)  # 20-80
train_index = [i for i in range(len(sklearn_x)) if i not in test_index]

# Test and train images
test_images = [sklearn_x[i] for i in test_index]
train_images = [sklearn_x[i] for i in train_index]

# Test and train targets
test_target = [sklearn_y[i] for i in test_index]
train_target = [sklearn_y[i] for i in train_index]

del test_index
del train_index
del sklearn_length
del digits
del sklearn_x
del sklearn_y
####################################################################

x_train_z = np.array(train_images)
y_train_z = np.array(train_target, dtype=np.uint8)

x_test_z = np.array(test_images)
y_test_z = np.array(test_target, dtype=np.uint8)

####################################################################

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(8, 8)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train_z, y_train_z, epochs=5)

loss, accuracy = model.evaluate(x_test_z, y_test_z)

print(accuracy)
print(loss)

model.save('digits.model')

for qqq in range(0, 10):
    img = cv.imread(f'8by8/{qqq}.png')[:, :, 0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f'The result is probably: {np.argmax(prediction)}')
    plt.imshow(img[0], cmap=plt.get_cmap('gray'))
    plt.show()
