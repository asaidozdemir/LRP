import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn import svm, datasets, neighbors
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
import numpy as np
import cv2 as cv

digits = load_digits()
n_samples = len(digits.images)

X = digits.images.reshape((n_samples, -1))
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y)

################################
del X
del y
del digits
del n_samples
################################

model_linear = svm.SVC(kernel='linear', degree=3, gamma='scale')
model_linear.fit(X_train, y_train)

#y_pred_linear = model_linear.predict(X_test)
model_linear.score(X_test, y_test)

# model_RBF = svm.SVC(degree=3, gamma='scale', kernel='rbf')
# model_RBF.fit(X_train, y_train)
#
# #y_pred_RBF = model_RBF.predict(X_test)
# model_RBF.score(X_test, y_test)

predictions = model_linear.predict(X_test)
print(classification_report(y_test, predictions))

for qqq in range(0, 10):
    img = cv.imread(f'8by8/{qqq}.png')[:, :, 0]
    img = np.invert(np.array([img]))
    prediction = model_linear.predict(img.reshape(1, -1))
    print(f'The result is probably: {np.max(prediction)}')
    print(f'The Image Was: {qqq}\n')
    plt.imshow(img[0], cmap=plt.get_cmap('gray'))
    plt.show()
