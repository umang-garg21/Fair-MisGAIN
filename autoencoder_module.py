
'''
Autoencoder module for categorical entries
'''

# Necessary packages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2


def autoencoder(data):
    no, dim = data.shape
    latent_dim = dim/2
    ## Building the autoencoder
    inputs = Input(dim, )
    e = Dense(no, activation="relu")(inputs)
    h = Dense(latent_dim, activation="relu")(e)
    d = Dense(no, activation="relu")(h)
    outputs = Dense(dim, activation="sigmoid")(d)

    autoencoder = Model(inputs, outputs)
    autoencoder.compile(optimizer=Adam(1e-3), loss='binary_crossentropy')
    autoencoder.summary()

    ## Training the autoencoder
    autoencoder.fit(
        data,
        data,
        epochs=50,
        batch_size=no,
        shuffle=False)

    auto_encoder_data = autoencoder.predict(data)

    return auto_encoder_data