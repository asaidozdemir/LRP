import time
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, neighbors
from sklearn.model_selection import train_test_split

# load Digits data set divided into data X and labels y
X, y = datasets.load_digits(return_X_y=True)

# splitting into train and test data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


def lets_knn(X_train, y_train, X_test, y_test, n_neighbors=3, weights='uniform', print_wrong_pred=False):
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
        img = cv.imread(f'8by8/{qqq}.png')[:, :, 0]
        img = np.invert(np.array([img]))
        prediction = knn.predict(img.reshape(1, -1))
        print(f'The result is probably: {np.max(prediction)}')
        print(f'The Image Was: {qqq}\n')
        plt.imshow(img[0], cmap=plt.get_cmap('gray'))
        plt.show()


lets_knn(X_train, y_train, X_test, y_test, 5, 'uniform', print_wrong_pred=True)
