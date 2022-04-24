import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn import svm
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

####################################################################

(x_train_new, y_train_new), (x_test_new, y_test_new) = mnist.load_data()

##################################################

train_count = 1000
test_count = 300

train_target_z = y_train_new[0:train_count].tolist()
flat_image = []
flat_array = []
for i in range(0, train_count):
    flat_image = np.ravel(x_train_new[i])
    flat_image = np.asarray(flat_image)
    flat_array.append(flat_image)
train_images_z = np.asarray(flat_array)
del flat_image
del flat_array
##################################################
test_target_z = y_test_new[0:test_count].tolist()
flat_image = []
flat_array = []
for i in range(0, test_count):
    flat_image = np.ravel(x_test_new[i])
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

del x_train_new
del x_test_new
del y_train_new
del y_test_new

del X
del y
del i
del digits
del n_samples
####################################################################

model_linear = svm.SVC(kernel='linear', degree=3, gamma='scale')
model_linear.fit(X_train, y_train)

# #y_pred_linear = model_linear.predict(X_test)
model_linear.score(X_test, y_test)

# model_RBF = svm.SVC(kernel='rbf', degree=3, gamma='scale')
# model_RBF.fit(X_train, y_train)
#
# #y_pred_RBF = model_RBF.predict(X_test)
# model_RBF.score(X_test, y_test)

predictions = model_linear.predict(X_test)
print(classification_report(y_test, predictions))

for qqq in range(0, 10):
    img = cv.imread(f'28by28/{qqq}.png')[:, :, 0]
    img = np.invert(np.array([img]))
    prediction = model_linear.predict(img.reshape(1, -1))
    print(f'The result is probably: {np.max(prediction)}')
    print(f'The Image Was: {qqq}\n')
    plt.imshow(img[0], cmap=plt.get_cmap('gray'))
    plt.show()
