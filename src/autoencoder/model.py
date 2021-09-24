from tensorflow.keras import layers, metrics
from tensorflow.keras.models import Model
import os
import logging 

def get_autoencoder(input_shape = (512,512,1)):
    i = layers.Input(shape=input_shape)

    x = layers.Conv2D(4, (7, 7))(i)
    x = layers.Conv2D(10, (5,5))(x)
    x = layers.Conv2D(20, (3,3))(x)
    x = layers.Conv2DTranspose(20, (3,3))(x)
    x = layers.Conv2DTranspose(10, (5,5))(x)
    x = layers.Conv2DTranspose(4, (7,7))(x)
    x = layers.Conv2DTranspose(1, (1,1))(x)
    return Model(i, x)

def get_compiled_model(input_shape = (512,512,1)):
    model = get_autoencoder(input_shape) 
    model.compile(optimizer="adadelta", loss="binary_crossentropy", metrics=[
                        metrics.MeanSquaredError()])
    return model

def get_compiled_model_with_weights(model_path, input_shape = (512,512,1)):
    model = get_compiled_model(input_shape)
    if (os.path.exists(model_path)):
        logging.info(f'Loading weights from \'{model_path}\'')
        model.load_weights(model_path)
    else:
        message = f'Model path was not found! ({model_path})'
        logging.fatal(message)
        raise FileNotFoundError(model_path)
    return model
    