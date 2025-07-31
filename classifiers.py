import numpy as np
from matplotlib import pyplot as plt
# from utils.EEGModels import EEGNet, TSGLEEGNet, DeepConvNet, ShallowConvNet, TSGLEEGNet

# import utils.variables as v
import matplotlib.pyplot as plt
import metrics as m

from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

import plotly.graph_objects as go
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from keras import layers
import keras.backend as K
import plotly.subplots as p


class NN(keras.Model):
    def __init__(self, input_size, output_size, num_layers, num_neurons, dropout_rate):
        super(NN, self).__init__()

        hidden_layers = []
        for i in range(num_layers):
            in_features = input_size if i == 0 else num_neurons[i - 1]
            out_features = num_neurons[i]
            hidden_layers.append(layers.Dense(out_features, activation='relu'))
            hidden_layers.append(layers.Dropout(dropout_rate))

        self.hidden_sequential = keras.Sequential(hidden_layers)
        self.fc_final = layers.Dense(output_size, activation='sigmoid')

    def call(self, x):
        """
        'call' adalah pengganti 'forward' di TensorFlow/Keras.
        """
        x = self.hidden_sequential(x)
        x = self.fc_final(x)
        return x


class EEG_CNN(keras.Model):
    def __init__(self, num_channels, num_classes):
        super(EEG_CNN, self).__init__()

        # Blok Konvolusi Pertama
        self.conv1 = layers.Conv2D(
            64, kernel_size=3, strides=1, activation='relu', padding='same')
        self.pool1 = layers.MaxPool2D(pool_size=2, strides=2)

        # Blok Konvolusi Kedua
        self.conv2 = layers.Conv2D(
            128, kernel_size=3, strides=1, activation='relu', padding='same')
        self.pool2 = layers.MaxPool2D(pool_size=2, strides=2)

        # Lapisan Flatten
        self.flat = layers.Flatten()

        # Lapisan Fully Connected (Dense)
        self.fc1 = layers.Dense(128)
        self.dropout1 = layers.Dropout(0.1)

        self.fc2 = layers.Dense(64)
        self.dropout2 = layers.Dropout(0.1)

        self.fc3 = layers.Dense(64)
        self.dropout3 = layers.Dropout(0.1)

        # Lapisan Output
        # num_classes harus 2 untuk one-hot
        self.fc4 = layers.Dense(num_classes)

    def call(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.flat(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        return x


class StressCapsuleLayer(layers.Layer):
    def __init__(self, num_capsules, capsule_dim, routing_iterations=3, **kwargs):
        super(StressCapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.routing_iterations = routing_iterations

    def build(self, input_shape):
        self.input_capsule_dim = input_shape[-1]
        num_input_capsules = input_shape[-2]
        self.W = self.add_weight(
            shape=(1, num_input_capsules, self.num_capsules,
                   self.capsule_dim, self.input_capsule_dim),
            initializer='glorot_uniform', name='W_transformation')

    def call(self, inputs):
        inputs_expanded = tf.expand_dims(inputs, axis=2)
        inputs_tiled = tf.tile(inputs_expanded, [1, 1, self.num_capsules, 1])
        inputs_tiled = tf.expand_dims(inputs_tiled, axis=4)

        prediction_vectors = tf.matmul(self.W, inputs_tiled)
        prediction_vectors = tf.squeeze(prediction_vectors, axis=4)

        routing_logits = tf.zeros(
            shape=(tf.shape(inputs)[0], 648, self.num_capsules, 1))

        for i in range(self.routing_iterations):
            coupling_coeffs = tf.nn.softmax(routing_logits, axis=2)
            weighted_sum = tf.reduce_sum(
                coupling_coeffs * prediction_vectors, axis=1, keepdims=True)
            squashed_outputs = self.squash(weighted_sum, axis=-1)
            if i < self.routing_iterations - 1:
                agreement = tf.reduce_sum(
                    prediction_vectors * squashed_outputs, axis=-1, keepdims=True)
                routing_logits += agreement
        return tf.squeeze(squashed_outputs, axis=1)

    def squash(self, vectors, axis=-1):
        s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
        scale = s_squared_norm / (1 + s_squared_norm) / \
            tf.sqrt(s_squared_norm + tf.keras.backend.epsilon())
        return scale * vectors

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_capsules, self.capsule_dim)


class CCN(keras.Model):
    def __init__(self, num_stress_classes=2, stress_capsule_dim=16, name='CCN_Model', **kwargs):
        super(CCN, self).__init__(name=name, **kwargs)

        # 1. Continuous Convolution Stage
        self.conv1 = layers.Conv2D(64, (2, 2), padding='same')
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.Activation('relu')

        self.conv2 = layers.Conv2D(128, (2, 2), padding='same')
        self.bn2 = layers.BatchNormalization()
        self.act2 = layers.Activation('relu')

        self.conv3 = layers.Conv2D(256, (2, 2), padding='same')
        self.bn3 = layers.BatchNormalization()
        self.act3 = layers.Activation('relu')

        self.conv4 = layers.Conv2D(64, (1, 1), padding='same')
        self.bn4 = layers.BatchNormalization()
        self.act4 = layers.Activation('relu')

        self.dropout = layers.Dropout(0.2)

        # 2. Primary Capsule Stage
        self.reshape_primary = layers.Reshape((648, 8))

        # 3. Stress Capsule Stage
        self.stress_capsule = StressCapsuleLayer(
            num_capsules=num_stress_classes,
            capsule_dim=stress_capsule_dim
        )

    def call(self, inputs, training=None):
        # Definisikan alur maju (forward pass)
        # Continuous Convolution
        # Teruskan argumen 'training' ke layer BatchNormalization dan Dropout
        x = self.act1(self.bn1(self.conv1(inputs), training=training))
        x = self.act2(self.bn2(self.conv2(x), training=training))
        x = self.act3(self.bn3(self.conv3(x), training=training))
        x = self.act4(self.bn4(self.conv4(x), training=training))
        x = self.dropout(x, training=training)

        # Primary Capsule
        x = self.reshape_primary(x)

        # Stress Capsule
        outputs = self.stress_capsule(x)

        return outputs


def knn_classification(train_data, test_data, train_labels, test_labels):
    param_grid = {'leaf_size': range(1, 10),
                  'n_neighbors': range(1, 5),
                  'p': [1, 2]}
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    knn_clf = GridSearchCV(KNeighborsClassifier(),
                           param_grid, refit=True, n_jobs=-1, cv=10)
    knn_clf.fit(train_data, train_labels)

    y_pred = knn_clf.predict(test_data)
    y_true = test_labels

    print(knn_clf.best_estimator_)
    print(knn_clf.best_params_)
    results = knn_clf.cv_results_
    print(results)

    # extract the relevant scores
    leaf_sizes = results['param_leaf_size'].data
    n_neighbors = results['param_n_neighbors'].data
    accuracies = results['mean_test_score']

    print('Number␣of␣results:', len(accuracies))
    # print('n_neighbors:', n_neighbors)
    # print('leaf_sizes:', leaf_sizes)
    print('overall␣accuracy:', np.round(
        np.sum(accuracies) / len(accuracies) * 100, 2))
    # plot the results
    plt.figure(1)
    plt.plot(range(len(accuracies)), accuracies)

    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.show()

    conf_matrix = metrics.confusion_matrix(y_true, y_pred)
    m.plot_conf_matrix_and_stats(conf_matrix)


def knn(data, label):
    X_train, X_test, y_train, y_test = train_test_split(
        data, label, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Gunakan parameter default yang cepat
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    if conf_matrix.shape != (2, 2):
        return [0, 0, 0]

    TN, FP, FN, TP = conf_matrix.ravel()
    accuracy = (TP + TN) / (TP + TN + FP +
                            FN) if (TP + TN + FP + FN) > 0 else 0
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    return [accuracy, sensitivity, specificity]


def svm_classification(train_data, test_data, train_labels, test_labels):
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }
    scaler = RobustScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    svm_clf = GridSearchCV(SVC(), param_grid, refit=True, n_jobs=-1, cv=10)
    svm_clf.fit(train_data, train_labels)

    y_pred = svm_clf.predict(test_data)
    y_true = test_labels

    print(svm_clf.best_estimator_)
    print(svm_clf.best_params_)

    # fit the grid search to get the results
    results = svm_clf.cv_results_
    print(results)

    # extract the relevant scores
    C_values = results['param_C'].data
    kernel_values = results['param_kernel'].data
    accuracies = results['mean_test_score']

    print('Number␣of␣results:', len(accuracies))
    # print('C_values:', C_values)
    # print('kernel_values:', kernel_values)
    print('overall␣accuracy:', np.round(
        np.sum(accuracies) / len(accuracies) * 100, 2))
    # plot the results
    plt.figure(2)
    plt.plot(
        range(len(accuracies)),
        accuracies
    )
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.show()

    conf_matrix = metrics.confusion_matrix(y_true, y_pred)
    m.plot_conf_matrix_and_stats(conf_matrix)
