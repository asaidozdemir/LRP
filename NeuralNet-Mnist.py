import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.datasets.mnist as mnist

# mnist = tf.keras.datasets.mnist
# (train_X, train_y), (test_X, test_y) = mnist.load_data()
# for i in range(9):
#     plt.subplot(330 + 1 + i)
#     plt.imshow(train_X[i], cmap=plt.get_cmap('gray'))
#     plt.show()


(x_train, y_train),(x_test, y_test)=mnist.load_data()

x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.softmax))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=5)

loss,accuracy=model.evaluate(x_test,y_test)

print(accuracy)
print(loss)

model.save('digits.model')

for x in range(0,10):
    img = cv.imread(f'28by28/{x}.png')[:,:,0]
    img = np.invert(np.array([img]))
    prediction= model.predict(img)
    print(f'The result is probably: {np.argmax(prediction)}')
    plt.imshow(img[0], cmap=plt.get_cmap('gray'))
    plt.show()