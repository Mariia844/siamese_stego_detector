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
from data_loading import ImageDataLoader
from eager_data import EagerDataGenerator, SiameseNetworkDataGenerator, SiameseStegoEagerDataGenerator, StegoData, TwoLegStegoDataGenerator
import ml_utils
import numpy as np

# Provided two tensors t1 and t2
# Euclidean distance = sqrt(sum(square(t1-t2)))
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

def get_model():    
    input = layers.Input((512,512,1))
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

    x = layers.Lambda(euclidean_distance)([tower_1, tower_2])
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    output_layer = layers.Dense(2, activation="sigmoid")(x)

    siamese = keras.Model(inputs=[input_1, input_2], outputs=output_layer)
    siamese.compile(optimizer="adadelta", loss=l1_loss(), metrics=[
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
    images_config = config['images']
    telegram_config = config['telegram']
    siamese_config = config['siamese']
    bot = telebot.TeleBot(telegram_config['token'])
    chat_id = telegram_config['chat_id']
    stego_path = siamese_config['stego_path']
    stego_dae_path = siamese_config['stego_dae_path']
    cover_path = siamese_config['cover_path']
    cover_dae_path = siamese_config['cover_dae_path']
    extension = siamese_config['extension']

    # paths_with_labels = [
    #     [[stego_path, stego_dae_path], [1]],
    #     [[cover_path, cover_dae_path], [0]],
    # ]
    # data_generator = SiameseNetworkDataGenerator(paths_with_labels, 16, False)
    # data_generator_1 = SiameseNetworkDataGenerator(paths_with_labels, 16, False)



    # + '\\' + '*' + extension

    stego_training_path = os.path.join(stego_path, 'training') 
    stego_dae_training_path = os.path.join(stego_dae_path, 'training') 
    cover_training_path = os.path.join(cover_path, 'training') 
    cover_dae_training_path = os.path.join(cover_dae_path, 'training') 
    
    stego_validation_path = os.path.join(stego_path, 'validation') 
    stego_dae_validation_path = os.path.join(stego_dae_path, 'validation') 
    cover_validation_path = os.path.join(cover_path, 'validation') 
    cover_dae_validation_path = os.path.join(cover_dae_path, 'validation') 

    stego_train_data = StegoData(cover_path=cover_training_path, cover_dae_path=cover_dae_training_path, stego_path=stego_training_path, stego_dae_path=stego_dae_training_path)
    stego_test_data = StegoData(cover_path=cover_validation_path, cover_dae_path=cover_dae_validation_path, stego_path=stego_validation_path, stego_dae_path=stego_dae_validation_path)

    data_loader = ImageDataLoader((512,512), 1)
    two_leg_train_generator = TwoLegStegoDataGenerator(stego_train_data, data_loader, 16, shuffle=True)
    two_leg_test_generator = TwoLegStegoDataGenerator(stego_test_data, data_loader, 16, shuffle=True)

    

    # stego_generator = SiameseStegoEagerDataGenerator(append_extension(stego_training_path, extension), 1, (512,512), 16, False)   
    # stego_dae_generator = SiameseStegoEagerDataGenerator(append_extension(stego_dae_training_path, extension), 1, (512,512), 16, False)
    # stego_combined_generator = combine_generators(stego_generator, stego_dae_generator, np.full(stego_generator.id_length, 1))
    # cover_generator = SiameseStegoEagerDataGenerator(append_extension(cover_training_path, extension), 0, (512,512), 16, False)
    # cover_dae_generator = SiameseStegoEagerDataGenerator(append_extension(cover_dae_training_path, extension), 0, (512,512), 16, False)
    # cover_combined_generator = combine_generators(cover_generator, cover_dae_generator, np.full(stego_generator.id_length, 0))
    # combined_generator = random_generator_choise(stego_combined_generator, cover_combined_generator, stego_generator.id_length)

    # stego_validation_generator = SiameseStegoEagerDataGenerator(append_extension(stego_validation_path, extension), 1, (512,512), 16, False)   
    # stego_dae_validation_generator = SiameseStegoEagerDataGenerator(append_extension(stego_dae_validation_path, extension), 1, (512,512), 16, False)
    # stego_combined_validation_generator = combine_generators(stego_validation_generator, stego_dae_validation_generator, np.full(stego_validation_generator.id_length, 1))
    # cover_validation_generator = SiameseStegoEagerDataGenerator(append_extension(cover_validation_path, extension), 0, (512,512), 16, False)
    # cover_dae_validation_generator = SiameseStegoEagerDataGenerator(append_extension(cover_dae_validation_path, extension), 0, (512,512), 16, False)
    # cover_combined_validation_generator = combine_generators(cover_validation_generator, cover_dae_validation_generator, np.full(stego_validation_generator.id_length, 0))
    # combined_validation_generator = random_generator_choise(stego_combined_validation_generator, cover_combined_validation_generator, stego_validation_generator.id_length)

    


    # next_batch = next(enumerate(combined_generator))[1]
    # next_two_legged_batch = next(enumerate(two_leg_train_generator))[1]
    # pass
    
    save_path = siamese_config['save_path']
    filepath = save_path + "/saved-model-ep_{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1,
        save_best_only=False, mode='auto', period=1)
    siamese = get_model()
    # history = siamese.fit(
    #     combined_generator, #two_leg_train_generator,
    #     validation_data = combined_validation_generator,#two_leg_test_generator,
    #     epochs = 10,
    #     batch_size = 16,
    #     shuffle = False,
    #     callbacks=[checkpoint])

    siamese.summary()

    # enumerator = enumerate(two_leg_train_generator)
    # next_value = None
    # while True:
    #     next_value = next(enumerator, None)
    #     if next_value == None:
    #         break
    #     if (next_value[1][0][0].shape == (0,)):
    #         break
    #     pass
    # return

    history = siamese.fit(
        two_leg_train_generator,
        validation_data = two_leg_test_generator,
        epochs = 150,
        batch_size = 16,
        shuffle = False,
        callbacks=[checkpoint])
    try:
        ml_utils.dump_object(history.history, os.path.join(save_path, 'history.dump'))
        
    except Exception as e:
        print('Save pickle ERROR: ', e)
    try:
        ml_utils.save_model_history_csv(history, os.path.join(save_path, 'history.csv'))
    except Exception as e:
        print('Save csv ERROR: ', e)

    bot.send_message(chat_id=chat_id, text=f'Siamese training completed')

    # stego_path = images_config['stego_path']
    # cover_path = images_config['cover_path']
    # save_path = images_config['save_path']
    # create_dir = bool(images_config['create_datetime_dir'])
    # load_model = bool(images_config['load_model'])
    # model_path = images_config['model_path']

if __name__ == "__main__":
    main()

