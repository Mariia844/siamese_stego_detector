from logging import debug
import os
from warnings import catch_warnings
import telebot
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, metrics
from tensorflow.python.framework.ops import reset_default_graph
from tensorflow.python.keras.metrics import TruePositives
from common import get_config
from tensorflow.python.keras.callbacks import ModelCheckpoint
from data_loading import ImageDataLoader, OpenCVImageDataLoader
from eager_data import EagerDataGenerator, SiameseNetworkDataGenerator, SiameseStegoEagerDataGenerator, StegoData, TwoLegStegoDataGenerator
import ml_utils
import numpy as np
import common

# Provided two tensors t1 and t2

def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """
    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

def l1_distance(vects):
    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.abs(x - y), axis=1, keepdims=True)
    return tf.math.abs(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

def l1_distance_v2(vects):
    x, y = vects
    return tf.math.abs(x - y)

def loss(margin=1):
    """Provides 'constrastive_loss' an enclosing scope with variable 'margin'.

  Arguments:
      margin: Integer, defines the baseline for distance for which pairs
              should be classified as dissimilar. - (default is 1).

  Returns:
      'constrastive_loss' function with data ('margin') attached.
  """

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
        """Calculates the constrastive loss.

      Arguments:
          y_true: List of labels, each label is of type float32.
          y_pred: List of predictions of same length as of y_true,
                  each label is of type float32.

      Returns:
          A tensor containing constrastive loss as floating point value.
      """
        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )

    return contrastive_loss

def l1_loss():
    def absolute_loss(y_pred, y_true):
        return tf.math.reduce_mean(
            tf.math.abs(y_true - y_pred)   
        )
    return absolute_loss

def get_siamese_network(input_shape : tuple = (512,512,1), dae_input: keras.Model = None) -> keras.Model:
    input = dae_input
    if input == None:
        input = layers.Input(input_shape)
    x = layers.Conv2D(16, (7, 7), activation="tanh", strides=(2,2))(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (5, 5), activation="tanh")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2,1), strides=(2,2))(x)
    x = layers.Conv2D(64, (3, 3), activation="tanh")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2,1), strides=(2,2))(x)
    x = layers.Conv2D(16, (1, 1), activation="tanh")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    embedding_network = keras.Model(input, x)


    input_1 = layers.Input((512,512,1))
    input_2 = layers.Input((512,512,1))

    # As mentioned above, Siamese Network share weights between
    # tower networks (sister networks). To allow this, we will use
    # same embedding network for both tower networks.
    tower_1 = embedding_network(input_1)
    tower_2 = embedding_network(input_2)

    x = layers.Lambda(l1_distance_v2)([tower_1, tower_2])
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    output_layer = layers.Dense(2, activation="sigmoid")(x)

    siamese = keras.Model(inputs=[input_1, input_2], outputs=output_layer)
    return siamese

def get_model(input_shape : tuple = (512,512,1)):    
    siamese = get_siamese_network(input_shape)
    siamese.compile(optimizer="adadelta", loss=loss(margin=0.5), metrics=[
                        metrics.MeanSquaredError(),
                        metrics.TruePositives(),
                        metrics.TrueNegatives(),
                        metrics.FalseNegatives(),
                        metrics.FalsePositives(),
                        metrics.AUC()])
    return siamese


def combine_generators(generator, dae_generator, labels, batch_size = 16):
    enumX1 = enumerate(generator)
    enumX2 = enumerate(dae_generator)
    current_index = 0
    while True:
        X1 = next(enumX1, None)
        X2 = next(enumX2, None)
        if X1 == None or X2 == None:
            break
        yield [X1[1], X2[1]], np.array(labels[current_index:current_index+batch_size]).astype('float32')
        current_index += batch_size

def random_generator_choise(gen1, gen2, length):
    import numpy as np
    choises = np.zeros(length).astype(int)
    choises = np.concatenate((choises, np.ones(length))).astype(int)
    np.random.shuffle(choises)
    enumerations = [enumerate(gen1), enumerate(gen2)]
    index = 0
    while True:
        choise = choises[index]
        index += 1
        enumeration = enumerations[choise]
        result = next(enumeration, None)
        if result == None:
            break
        result = result[1]
        yield result

def append_extension(path, extension):
    return os.path.join(path, extension)

def main():
    config = get_config()
    telegram_config = config['telegram']
    siamese_config = config['siamese']
    stego_path = siamese_config['stego_path']
    stego_dae_path = siamese_config['stego_dae_path']
    cover_path = siamese_config['cover_path']
    cover_dae_path = siamese_config['cover_dae_path']

    extension = siamese_config['extension']
    stego_training_path = os.path.join(stego_path, 'training') 
    stego_dae_training_path = os.path.join(stego_dae_path, 'training') 
    cover_training_path = os.path.join(cover_path, 'training') 
    cover_dae_training_path = os.path.join(cover_dae_path, 'training') 
    
    stego_validation_path = os.path.join(stego_path, 'validation') 
    stego_dae_validation_path = os.path.join(stego_dae_path, 'validation') 
    cover_validation_path = os.path.join(cover_path, 'validation') 
    cover_dae_validation_path = os.path.join(cover_dae_path, 'validation') 

    stego_train_data = StegoData(
        cover_path=cover_training_path,
        cover_dae_path=cover_dae_training_path, 
        stego_path=stego_training_path, 
        stego_dae_path=stego_dae_training_path,
        types=[extension])
    stego_test_data = StegoData(
        cover_path=cover_validation_path, 
        cover_dae_path=cover_dae_validation_path, 
        stego_path=stego_validation_path, 
        stego_dae_path=stego_dae_validation_path,
        types=[extension])

    # data_loader = OpenCVImageDataLoader((512,512), 1)
    data_loader = ImageDataLoader((512,512), 1) 
    two_leg_train_generator = TwoLegStegoDataGenerator(stego_train_data, data_loader, 16, shuffle=True)
    two_leg_test_generator = TwoLegStegoDataGenerator(stego_test_data, data_loader, 16, shuffle=True)
    
    save_path = siamese_config['save_path']
    filepath = save_path + "/saved-model-ep_{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1,
        save_best_only=False, mode='auto', period=1)
    # print_callback = ml_utils.PrintCallback()
    save_callback = ml_utils.SaveStatsCallback(history_path=os.path.join(save_path, 'history.csv'))
    siamese = get_model()
    # siamese.summary()
    history = siamese.fit(
        two_leg_train_generator,
        validation_data = two_leg_test_generator,
        epochs = 400,
        batch_size = 16,
        shuffle = True,
        callbacks=[checkpoint, save_callback])
    import win32file
    win32file._setmaxstdio(2048)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    try:
        ml_utils.save_model_history_csv(history, os.path.join(save_path, 'history_full.csv'))
    except Exception as e:
        print('Save csv ERROR: ', e)

    common.send_message(text=f'Siamese training completed')

if __name__ == "__main__":
    main()

