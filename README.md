This is the documentation for "Optical Character Recognition with Machine Learning"

This collection is made for Python Interpreter version 3.8 

The Required Python Libraries to run the algorithms are:

-matplotlib
-keras
-tensorflow
-opencv-python
-scikit-learn
-numpy

----------------------------------------------------------------------------------------------

DATASET Features
1-) SKLearn Digits Dataset = 1797 (8*8 array) (0-15 white/black level)
2-) MNist Digits Dataset = 70000 (28*28 array) (0-255 white/black level)

----------------------------------------------------------------------------------------------

ALGORITHM Features

- KNN
n_neighbors     = 3             / Number of neighbors
weights         = uniform       / {uniform, distance}
algorithm       = auto          / {auto, ball_tree, kd_tree, brute}

- SVM
kernel          = linear        / {linear, poly, rbf, sigmoid, precomputed}
degree          = 3             / Degree of the polynomial kernel
gamma           = scale         / {scale, auto} Kernel coefficient

- NN
layers          = 3             / {linear, poly, rbf, sigmoid, precomputed}
optimizer       = adam          / {SGD,RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl}
epochs          = 5             / default=3
batch_size      = default=32    / 1875*32 = 60.000 image (for Mnist)
class_weight    = default=None
sample_weight   = default=None
loss            = sparse_categorical_crossentropy

- RFC
sample_weight           = None          / Must be equal size weight array as samples
n_estimators            = default=100   / The number of trees in the forest.
max_depth               = default=None  / The maximum depth of the tree.
decision_function_shape =default='ovr   / {'ovo', 'ovr'} one-vs-one, one-vs-rest
