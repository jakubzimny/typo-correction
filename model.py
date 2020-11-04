from tensorflow import keras
from tensorflow.keras import layers

def autoencoder(input_shape) -> keras.Model:
    _, x = input_shape

    input_layer = keras.Input(shape=input_shape)
    encoder_1 = layers.Dense(x - 5, activation='relu')(input_layer)
    encoded_2 = layers.Dense(x - 15, activation='relu')(encoder_1)
    encoded_3 = layers.Dense(x - 20, activation='relu')(encoded_2)

    bottleneck = layers.Dense(1000, activation='relu')(encoded_3) 

    decoded_1 = layers.Dense(x - 20, activation='relu')(bottleneck)
    decoded_2 = layers.Dense(x - 15, activation='relu')(decoded_1)
    decoded_3 = layers.Dense(x - 10, activation='relu')(decoded_2)

    out = layers.Dense(x, activation='sigmoid')(decoded_3)

    autoencoder = keras.Model(input_layer, out)
    return autoencoder
