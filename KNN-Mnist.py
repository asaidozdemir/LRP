from sklearn import datasets, neighbors
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import time

# load Digits data set divided into data X and labels y
X, y = datasets.load_digits(return_X_y=True)

# splitting into train and test data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

####################################################################

(x_train, y_train_new), (x_test, y_test_new) = mnist.load_data()

##################################################
train_target_z = y_train_new.tolist()
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
test_target_z = y_test_new.tolist()
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

X_train = train_images_z
X_test = test_images_z

y_train = train_target_z
y_test = test_target_z

##################################################
del train_images_z
del test_images_z
del train_target_z
del test_target_z

del x_train
del x_test
del y_train_new
del y_test_new

del X
del y
del i


####################################################################


def lets_knn(X_train, y_train, X_test, y_test, n_neighbors=3, weights='uniform'):
    t0 = time.time()
    # creating and training knn classifier
    knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    knn.fit(X_train, y_train)
    t1 = time.time()

    # predicting classes and comparing them with actual labels
    pred = knn.predict(X_test)
    print(type(pred))
    t2 = time.time()
    # calculating accuracy
    accuracy = round(np.mean(pred == y_test) * 100, 1)

    print("Accuracy of", weights, "KNN with", n_neighbors, "neighbors:", accuracy, "%. Fit in",
          round(t1 - t0, 1), "s. Prediction in", round(t2 - t1, 1), "s")

    for qqq in range(0, 10):
        img = cv.imread(f'28by28/{qqq}.png')[:, :, 0]
        img = np.invert(np.array([img]))
        prediction = knn.predict(img.reshape(1, -1))
        print(f'The result is probably: {np.max(prediction)}')
        print(f'The Image Was: {qqq}\n')
        plt.imshow(img[0], cmap=plt.get_cmap('gray'))
        plt.show()


lets_knn(X_train, y_train, X_test, y_test, 5, 'uniform')
