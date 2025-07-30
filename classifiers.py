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


# class SimpleNN(torch.nn.Module):
#     def __init__(self, input_size, output_size, num_layers = num_layers, num_neurons=num_neurons, dropout_rate=dropout_rate):
#         super(SimpleNN, self).__init__()
#         layers = []
#         for i in range(num_layers):
#             in_features = input_size if i == 0 else num_neurons[i - 1]
#             out_features = num_neurons[i]
#             layers.append(torch.nn.Linear(in_features, out_features))
#             layers.append(torch.nn.ReLU())
#             layers.append(torch.nn.Dropout(dropout_rate))
#         self.layers = torch.nn.Sequential(*layers)
#         self.fc_final = torch.nn.Linear(num_neurons[-1], output_size)
#         self.sigmoid = torch.nn.Sigmoid()

#     def forward(self, x):
#         x = self.layers(x)
#         x = self.fc_final(x)
#         x = self.sigmoid(x)
#         return x

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


# class EEG_CNN(torch.nn.Module):
#     def __init__(self, num_channels, num_classes):
#         super(EEG_CNN, self).__init__()

#         self.conv1 = torch.nn.Conv2d(
#             num_channels, 64, kernel_size=10, stride=1)
#         self.relu1 = torch.nn.ReLU()
#         self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

#         self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=10, stride=1)
#         self.relu2 = torch.nn.ReLU()
#         self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

#         self.flat = torch.nn.Flatten()

#         self.fc1 = torch.nn.Linear(5760, 128)
#         self.dropout1 = torch.nn.Dropout(0.1)

#         self.fc2 = torch.nn.Linear(128, 64)
#         self.dropout2 = torch.nn.Dropout(0.1)

#         self.fc3 = torch.nn.Linear(64, 64)
#         self.dropout3 = torch.nn.Dropout(0.1)

#         self.fc4 = torch.nn.Linear(64, num_classes)

#         # self.sigmoid = torch.nn.Sigmoid()

#     def forward(self, x):
#         x = self.pool1(self.relu1(self.conv1(x)))
#         x = self.pool2(self.relu2(self.conv2(x)))
#         x = self.flat(x)
#         x = self.fc1(x)
#         x = self.dropout1(x)
#         x = self.fc2(x)
#         x = self.dropout2(x)
#         x = self.fc3(x)
#         x = self.dropout3(x)
#         x = self.fc4(x)
#         # x = self.sigmoid(x)
#         return x


class EEG_CNN(keras.Model):
    def __init__(self, num_channels, num_classes):
        super(EEG_CNN, self).__init__()

        # Blok Konvolusi Pertama
        self.conv1 = layers.Conv2D(
            64, kernel_size=10, strides=1, activation='relu')
        self.pool1 = layers.MaxPool2D(pool_size=2, strides=2)

        # Blok Konvolusi Kedua
        self.conv2 = layers.Conv2D(
            128, kernel_size=10, strides=1, activation='relu')
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
        # ... (isi fungsi call sama persis seperti kode Anda)
        inputs_expanded = K.expand_dims(inputs, axis=2)
        inputs_tiled = K.tile(inputs_expanded, [1, 1, self.num_capsules, 1])
        inputs_tiled = K.expand_dims(inputs_tiled, axis=4)
        prediction_vectors = tf.matmul(self.W, inputs_tiled)
        prediction_vectors = K.squeeze(prediction_vectors, axis=4)
        routing_logits = tf.zeros(
            shape=(K.shape(inputs)[0], 648, self.num_capsules, 1))
        for i in range(self.routing_iterations):
            coupling_coeffs = K.softmax(routing_logits, axis=2)
            weighted_sum = tf.reduce_sum(
                coupling_coeffs * prediction_vectors, axis=1, keepdims=True)
            squashed_outputs = self.squash(weighted_sum, axis=-1)
            if i < self.routing_iterations - 1:
                agreement = tf.reduce_sum(
                    prediction_vectors * squashed_outputs, axis=-1, keepdims=True)
                routing_logits += agreement
        return K.squeeze(squashed_outputs, axis=1)

    def squash(self, vectors, axis=-1):
        # ... (isi fungsi squash sama persis seperti kode Anda)
        s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
        scale = s_squared_norm / (1 + s_squared_norm) / \
            K.sqrt(s_squared_norm + K.epsilon())
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

    def call(self, inputs):
        # Definisikan alur maju (forward pass)
        # Continuous Convolution
        x = self.act1(self.bn1(self.conv1(inputs)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.act4(self.bn4(self.conv4(x)))
        x = self.dropout(x)

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
    """Fungsi KNN sederhana untuk evaluasi fitness di Algoritma Genetika."""
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


# def svm_classification_SAM40(train_data, test_data, SAM40_data, train_labels, test_labels, SAM40_labels):
#     param_grid = {
#         'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000],
#         'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
#     }
#     print('Scaling␣training␣and␣testing␣data')
#     scaler = RobustScaler()
#     train_data = scaler.fit_transform(train_data)
#     test_data = scaler.transform(test_data)

#     print('Scaling␣SAM40␣data')
#     SAM40_scaler = RobustScaler()
#     SAM40_data = SAM40_scaler.fit_transform(SAM40_data)

#     print('Finding␣the␣best␣model')
#     svm_clf = GridSearchCV(SVC(), param_grid, refit=True, n_jobs=-1, cv=10)
#     svm_clf.fit(train_data, train_labels)

#     print(svm_clf.best_estimator_)
#     print(svm_clf.best_params_)

#     print('Predicting␣on␣test␣data')
#     y_pred = svm_clf.predict(test_data)
#     y_true = test_labels

#     # fit the grid search to get the results
#     results = svm_clf.cv_results_

#     # extract the relevant scores
#     C_values = results['param_C'].data
#     kernel_values = results['param_kernel'].data
#     accuracies = results['mean_test_score']

#     print('Number␣of␣results:', len(accuracies))
#     # print(’C_values:’, C_values)
#     # print(’kernel_values:’, kernel_values)
#     print('accuracies:', accuracies)
#     # plot the results
#     plt.figure(2)
#     plt.plot(
#         range(len(accuracies)), accuracies
#     )
#     plt.xlabel('Iteration')
#     plt.ylabel('Accuracy')
#     plt.show()

#     conf_matrix = metrics.confusion_matrix(y_true, y_pred)
#     m.plot_conf_matrix_and_stats(conf_matrix)

#     # SAM40
#     print('Predicting␣on␣SAM40␣data')
#     y_pred_SAM40 = svm_clf.predict(SAM40_data)
#     y_true_SAM40 = SAM40_labels
#     conf_matrix_SAM40 = metrics.confusion_matrix(y_true_SAM40, y_pred_SAM40)
#     m.plot_conf_matrix_and_stats(conf_matrix_SAM40)


# def kfold_EEGNet_classification(train_data, test_data, train_labels, test_labels, n_folds, data_type, epoched=True):
#     if epoched:
#         if data_type == 'new_ica':
#             model = EEGNet(nb_classes=2, Chans=v.NUM_CHANNELS,
#                            Samples=v.EPOCH_LENGTH*v.NEW_SFREQ,
#                            dropoutRate=0.5, kernLength=32, F1=8,
#                            D=2, F2=16, dropoutType='Dropout')
#         else:
#             model = EEGNet(nb_classes=2, Chans=v.NUM_CHANNELS,
#                            Samples=v.EPOCH_LENGTH*v.SFREQ,
#                            dropoutRate=0.5, kernLength=32, F1=8,
#                            D=2, F2=16, dropoutType='Dropout')
#     else:  # if not epoched
#         if data_type == 'new_ica':
#             model = EEGNet(nb_classes=2, Chans=v.NUM_CHANNELS,
#                            Samples=v.NEW_NUM_SAMPLES,
#                            dropoutRate=0.5, kernLength=32, F1=8,
#                            D=2, F2=16, dropoutType='Dropout')
#         else:
#             model = EEGNet(nb_classes=2, Chans=v.NUM_CHANNELS,
#                            Samples=v.NUM_SAMPLES,
#                            dropoutRate=0.5, kernLength=32, F1=8,
#                            D=2, F2=16, dropoutType='Dropout')

#     model.compile(loss='sparse_categorical_crossentropy',
#                   optimizer='adam', metrics=['accuracy'])

#     numParams = model.count_params()
#     checkpointer = ModelCheckpoint(
#         filepath='/tmp/checkpoint.h5', verbose=1, save_best_only=True)

#     class_weights = {0: 1, 1: 3}

#     # Split into k-folds
#     skf = StratifiedKFold(n_splits=n_folds)
#     total_accuracy = 0

#     for fold, (train_index, val_index) in enumerate(skf.split(train_data, train_labels)):
#         print(f"\nFold␣nr:␣{fold+1}")
#         train_data_fold = train_data[train_index]
#         train_labels_fold = train_labels[train_index]
#         val_data_fold = train_data[val_index]
#         val_labels_fold = train_labels[val_index]

#         history = model.fit(train_data_fold, train_labels_fold, batch_size=None, epochs=30, verbose=2, validation_data=(
#             val_data_fold, val_labels_fold), callbacks=[checkpointer], class_weight=class_weights)

#         # load optimal weights
#         model.load_weights('/tmp/checkpoint.h5')

#         probs = model.predict(test_data)
#         preds = probs.argmax(axis=-1)
#         conf_matrix = metrics.confusion_matrix(test_labels, preds)
#         m.plot_conf_matrix_and_stats(conf_matrix)

#         # Plot Loss/Accuracy over time
#         # Create figure with secondary y-axis
#         fig = p.make_subplots(specs=[[{"secondary_y": True}]])
#         # Add traces
#         fig.add_trace(go.Scatter(
#             y=history.history['val_loss'], name="val_loss"), secondary_y=False)
#         fig.add_trace(go.Scatter(
#             y=history.history['loss'], name="loss"), secondary_y=False)
#         fig.add_trace(go.Scatter(
#             y=history.history['val_accuracy'], name="val␣accuracy"), secondary_y=True)
#         fig.add_trace(go.Scatter(
#             y=history.history['accuracy'], name="accuracy"), secondary_y=True)

#         # Add figure title
#         fig.update_layout(title_text="Loss/Accuracy of k-folds EEGNet")
#         # Set x-axis title
#         fig.update_xaxes(title_text="Epoch")
#         # Set y-axes titles
#         fig.update_yaxes(title_text="Loss", secondary_y=False)
#         fig.update_yaxes(title_text="Accuracy", secondary_y=True)
#         fig.show()


# def kfold_TSGL_classification(train_data, test_data, train_labels, test_labels, n_folds, data_type, epoched=True):
#     if epoched:
#         if data_type == 'new_ica':
#             model = TSGLEEGNet(nb_classes=2, Chans=v.NUM_CHANNELS, Samples=v.EPOCH_LENGTH * v.NEW_SFREQ,
#                                dropoutRate=0.5, kernLength=128, F1=96, D=1, F2=96, dropoutType='Dropout')
#         else:
#             model = TSGLEEGNet(nb_classes=2, Chans=v.NUM_CHANNELS, Samples=v.EPOCH_LENGTH *
#                                v.SFREQ, dropoutRate=0.5, kernLength=128, F1=96, D=1, F2=96, dropoutType='Dropout')
#     else:  # if not epoched
#         if data_type == 'new_ica':
#             model = TSGLEEGNet(nb_classes=2, Chans=v.NUM_CHANNELS, Samples=v.NEW_NUM_SAMPLES,
#                                dropoutRate=0.5, kernLength=128, F1=96, D=1, F2=96, dropoutType='Dropout')
#         else:
#             model = TSGLEEGNet(nb_classes=2, Chans=v.NUM_CHANNELS, Samples=v.NUM_SAMPLES,
#                                dropoutRate=0.5, kernLength=128, F1=96, D=1, F2=96, dropoutType='Dropout')

#     # compile the model and set the optimizers
#     model.compile(loss='sparse_categorical_crossentropy',
#                   optimizer='adam', metrics=['accuracy'])

#     # count number of parameters in the model
#     numParams = model.count_params()

#     # set a valid path for your system to record model checkpoints
#     checkpointer = ModelCheckpoint(
#         filepath='/tmp/checkpoint.h5', verbose=1, save_best_only=True)
#     class_weights = {0: 1, 1: 3}

#     # Split into k-folds
#     skf = StratifiedKFold(n_splits=n_folds)
#     for fold, (train_index, val_index) in enumerate(skf.split(train_data, train_labels)):
#         print(f"\nFold␣nr:␣{fold+1}")
#         train_data_fold = train_data[train_index]
#         train_labels_fold = train_labels[train_index]
#         val_data_fold = train_data[val_index]
#         val_labels_fold = train_labels[val_index]

#         history = model.fit(train_data_fold, train_labels_fold, batch_size=None, epochs=30, verbose=2, validation_data=(
#             val_data_fold, val_labels_fold), callbacks=[checkpointer], class_weight=class_weights)

#     # load optimal weights
#     model.load_weights('/tmp/checkpoint.h5')

#     probs = model.predict(test_data)
#     preds = probs.argmax(axis=-1)

#     conf_matrix = metrics.confusion_matrix(test_labels, preds)
#     m.plot_conf_matrix_and_stats(conf_matrix)

#     # Plot Loss/Accuracy over time
#     # Create figure with secondary y-axis
#     fig = p.make_subplots(specs=[[{"secondary_y": True}]])
#     # Add traces
#     fig.add_trace(go.Scatter(
#         y=history.history['val_loss'], name="val_loss"), secondary_y=False)
#     fig.add_trace(go.Scatter(
#         y=history.history['loss'], name="loss"), secondary_y=False)
#     fig.add_trace(go.Scatter(
#         y=history.history['val_accuracy'], name="val␣accuracy"), secondary_y=True)
#     fig.add_trace(go.Scatter(
#         y=history.history['accuracy'], name="accuracy"), secondary_y=True)

#     # Add figure title
#     fig.update_layout(title_text="Loss/Accuracy of k-folds EEGNet")
#     # Set x-axis title
#     fig.update_xaxes(title_text="Epoch")
#     # Set y-axes titles
#     fig.update_yaxes(title_text="Loss", secondary_y=False)
#     fig.update_yaxes(title_text="Accuracy", secondary_y=True)
#     fig.show()


# def kfold_DeepConvNet_classification(train_data, test_data, train_labels, test_labels, n_folds, data_type, epoched=True):
#     if epoched:
#         if data_type == 'new_ica':
#             model = DeepConvNet(nb_classes=2, Chans=v.NUM_CHANNELS,
#                                 Samples=v.EPOCH_LENGTH * v.NEW_SFREQ,
#                                 dropoutRate=0.5)
#         else:
#             model = DeepConvNet(nb_classes=2, Chans=v.NUM_CHANNELS,
#                                 Samples=v.EPOCH_LENGTH * v.SFREQ,
#                                 dropoutRate=0.5)
#     else:  # if not epoched
#         if data_type == 'new_ica':
#             model = DeepConvNet(nb_classes=2, Chans=v.NUM_CHANNELS,
#                                 Samples=v.NEW_NUM_SAMPLES,
#                                 dropoutRate=0.5)
#         else:
#             model = DeepConvNet(nb_classes=2, Chans=v.NUM_CHANNELS,
#                                 Samples=v.NUM_SAMPLES,
#                                 dropoutRate=0.5)

#     # compile the model and set the optimizers
#     model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
#                   metrics=['accuracy'])

#     # count number of parameters in the model
#     numParams = model.count_params()

#     # set a valid path for your system to record model checkpoints
#     checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
#                                    save_best_only=True)

#     class_weights = {0: 1, 1: 3}

#     # Split into k-folds
#     skf = StratifiedKFold(n_splits=n_folds)
#     for fold, (train_index, val_index) in enumerate(
#             skf.split(train_data, train_labels)):
#         print(f"\nFold␣nr:␣{fold+1}")
#         train_data_fold = train_data[train_index]
#         train_labels_fold = train_labels[train_index]
#         val_data_fold = train_data[val_index]
#         val_labels_fold = train_labels[val_index]

#         history = model.fit(train_data_fold, train_labels_fold, batch_size=None,
#                             epochs=30, verbose=2, validation_data=(val_data_fold, val_labels_fold), callbacks=[checkpointer], class_weight=class_weights)

#     # load optimal weights
#     model.load_weights('/tmp/checkpoint.h5')
#     probs = model.predict(test_data)
#     preds = probs.argmax(axis=-1)

#     conf_matrix = metrics.confusion_matrix(test_labels, preds)
#     m.plot_conf_matrix_and_stats(conf_matrix)

#     # Plot Loss/Accuracy over time
#     #  Create figure with secondary y-axis
#     fig = p.make_subplots(specs=[[{"secondary_y": True}]])
#     # Add traces
#     fig.add_trace(go.Scatter(y=history.history['val_loss'], name="val_loss"),
#                   secondary_y=False)
#     fig.add_trace(go.Scatter(y=history.history['loss'], name="loss"),
#                   secondary_y=False)
#     fig.add_trace(go.Scatter(y=history.history['val_accuracy'],
#                              name="val␣accuracy"), secondary_y=True)
#     fig.add_trace(go.Scatter(y=history.history['accuracy'],
#                              name="accuracy"), secondary_y=True)

#     # Add figure title
#     fig.update_layout(title_text="Loss/Accuracy of k-folds EEGNet")
#     # Set x-axis title
#     fig.update_xaxes(title_text="Epoch")
#     # Set y-axes titles
#     fig.update_yaxes(title_text="Loss", secondary_y=False)
#     fig.update_yaxes(title_text="Accuracy", secondary_y=True)
#     fig.show()


# def kfold_ShallowConvNet_classification(train_data, test_data, train_labels,
#                                         test_labels, n_folds, data_type,
#                                         epoched=True):
#     if epoched:
#         if data_type == 'new_ica':
#             model = ShallowConvNet(nb_classes=2, Chans=v.NUM_CHANNELS,
#                                    Samples=v.EPOCH_LENGTH * v.NEW_SFREQ,
#                                    dropoutRate=0.5)
#         else:
#             model = ShallowConvNet(nb_classes=2, Chans=v.NUM_CHANNELS,
#                                    Samples=v.EPOCH_LENGTH*v.SFREQ,
#                                    dropoutRate=0.5)
#     else:  # if not epoched
#         if data_type == 'new_ica':
#             model = ShallowConvNet(nb_classes=2, Chans=v.NUM_CHANNELS,
#                                    Samples=v.NEW_NUM_SAMPLES,
#                                    dropoutRate=0.5)
#         else:
#             model = ShallowConvNet(nb_classes=2, Chans=v.NUM_CHANNELS,
#                                    Samples=v.NUM_SAMPLES,
#                                    dropoutRate=0.5)

#     # compile the model and set the optimizers
#     model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
#                   metrics=['accuracy'])
#     # count number of parameters in the model
#     numParams = model.count_params()
#     # set a valid path for your system to record model checkpoints
#     checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
#                                    save_best_only=True)

#     class_weights = {0: 1, 1: 3}

#     # Split into k-folds
#     skf = StratifiedKFold(n_splits=n_folds)
#     for fold, (train_index, val_index) in enumerate(skf.split(
#             train_data, train_labels)):
#         print(f"\nFold␣nr:␣{fold+1}")
#         train_data_fold = train_data[train_index]
#         train_labels_fold = train_labels[train_index]
#         val_data_fold = train_data[val_index]
#         val_labels_fold = train_labels[val_index]

#         history = model.fit(train_data_fold, train_labels_fold, batch_size=None,
#                             epochs=30, verbose=2, validation_data=(val_data_fold, val_labels_fold), callbacks=[checkpointer], class_weight=class_weights)

#     # load optimal weights
#     model.load_weights('/tmp/checkpoint.h5')
#     probs = model.predict(test_data)
#     preds = probs.argmax(axis=-1)

#     conf_matrix = metrics.confusion_matrix(test_labels, preds)
#     m.plot_conf_matrix_and_stats(conf_matrix)

#     # Plot Loss/Accuracy over time
#     # Create figure with secondary y-axis
#     fig = p.make_subplots(specs=[[{"secondary_y": True}]])
#     # Add traces
#     fig.add_trace(go.Scatter(y=history.history['val_loss'], name="val_loss"),
#                   secondary_y=False)
#     fig.add_trace(go.Scatter(y=history.history['loss'], name="loss"),
#                   secondary_y=False)
#     fig.add_trace(go.Scatter(y=history.history['val_accuracy'],
#                              name="val␣accuracy"), secondary_y=True)
#     fig.add_trace(go.Scatter(y=history.history['accuracy'], name="accuracy"),
#                   secondary_y=True)
#     # Add figure title
#     fig.update_layout(title_text="Loss/Accuracy of k-folds EEGNet")
#     # Set x-axis title
#     fig.update_xaxes(title_text="Epoch")
#     # Set y-axes titles
#     fig.update_yaxes(title_text="Loss", secondary_y=False)
#     fig.update_yaxes(title_text="Accuracy", secondary_y=True)
#     fig.show()
