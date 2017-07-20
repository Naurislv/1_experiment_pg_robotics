"""Simple Keras fully connected network with 3 layers

Simple guide: https://elitedatascience.com/keras-tutorial-deep-learning-in-python
"""

from keras.layers import Dense, Flatten, Convolution2D
from keras.models import Sequential
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import RMSprop
import tensorflow as tf

TF_CONFIG = tf.ConfigProto()
TF_CONFIG.gpu_options.per_process_gpu_memory_fraction = 0.6
set_session(tf.Session(config=TF_CONFIG))

class SimpleNet(object):
    """Three Dense layers"""

    def __init__(self, input_shape):

        self.model = Sequential()
        self.model.add(Convolution2D(24, (4, 4), padding='valid', activation='selu',
                                     strides=(2, 2), input_shape=input_shape))
        self.model.add(Convolution2D(36, (4, 4), padding='valid', activation='selu',
                                     strides=(2, 2)))
        self.model.add(Convolution2D(48, (4, 4), padding='valid', activation='selu',
                                     strides=(2, 2)))
        self.model.add(Convolution2D(64, (2, 2), padding='valid', activation='selu'))
        self.model.add(Convolution2D(64, (2, 2), padding='valid', activation='selu'))
        self.model.add(Flatten())
        self.model.add(Dense(100, activation='selu'))
        self.model.add(Dense(50, activation='selu'))
        self.model.add(Dense(10, activation='selu'))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy',  # binary classification
                           optimizer=RMSprop(lr=1e-4, decay=0.99),
                           metrics=['accuracy'])

        self.model.summary()

    def fit(self, observation, y_train):
        """Train neural network."""

        self.model.fit(observation, y_train, batch_size=32, nb_epoch=1, verbose=1)

    def predict(self, observation):
        """Predict action from observation."""
        return self.model.predict(observation)
