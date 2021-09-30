from autoencoder.model import get_autoencoder
from tensorflow.keras import metrics
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, MaxPool2D, Flatten, Lambda, Dense, Dropout

from data_loading import ImageDataLoader
from eager_data import EagerDataGenerator

import glob

DAE_WEIGHTS_PATH = "E:/Mary/SIAMESE_VECTOR/DAE_training_suni_0.5/20_09_2021_09_40_12/saved-model-ep_300-loss_0.59685.hdf5"
IMAGES_PATH = "E:/Mary/SIAMESE_VECTOR/siamese_data/suni_0.5/cover_1k/training/*.pgm"
IMAGE_SIZE = 512


FILES = glob.glob(IMAGES_PATH)

dae_model = get_autoencoder()
dae_model.trainable = False
dae_model.load_weights(DAE_WEIGHTS_PATH)



def l1_distance_v2(vects):
    x, y = vects
    return tf.math.abs(x - y)


def siamese_loss(margin=1):
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


def get_siamese_network(input_shape : tuple = (IMAGE_SIZE,IMAGE_SIZE,1)) -> Model:
    
    input = Input(input_shape)
        
    x1 = Conv2D(16, (7, 7), activation="tanh", strides=(2,2))(input)
    x1 = BatchNormalization()(x1)
    x1 = Conv2D(32, (5, 5), activation="tanh")(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPool2D((2,1), strides=(2,2))(x1)
    x1 = Conv2D(64, (3, 3), activation="tanh")(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPool2D((2,1), strides=(2,2))(x1)
    x1 = Conv2D(16, (1, 1), activation="tanh")(x1)
    x1 = BatchNormalization()(x1)
    x1 = Flatten()(x1)
    embedding_network_1 = Model(input, x1)
    

    # x1 = Lambda(dae_output)(input)
    x1 = dae_model(input)
    x1 = Conv2D(16, (7, 7), activation="tanh", strides=(2,2))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Conv2D(32, (5, 5), activation="tanh")(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPool2D((2,1), strides=(2,2))(x1)
    x1 = Conv2D(64, (3, 3), activation="tanh")(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPool2D((2,1), strides=(2,2))(x1)
    x1 = Conv2D(16, (1, 1), activation="tanh")(x1)
    x1 = BatchNormalization()(x1)
    x1 = Flatten()(x1)
    embedding_network_2= Model(input, x1)

    input_1 = Input(input_shape)

    tower_1 = embedding_network_1(input_1)
    tower_2 = embedding_network_2(input_1)

    # L1 distance layer
    x = Lambda(l1_distance_v2)([tower_1, tower_2])
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.2)(x)
    output_layer = Dense(2, activation="sigmoid")(x)

    siamese = Model(inputs=[input_1], outputs=output_layer)
    return siamese

siamese = get_siamese_network()
siamese.compile(optimizer="adadelta", loss=siamese_loss(margin=0.5), metrics=[
                        metrics.MeanSquaredError(),
                        metrics.TruePositives(),
                        metrics.TrueNegatives(),
                        metrics.FalseNegatives(),
                        metrics.FalsePositives(),
                        metrics.AUC()])

generator = EagerDataGenerator(FILES, ImageDataLoader(IMAGE_SIZE, 1))
next_batch = next(iter(generator))

predictions = siamese.predict(next_batch)
print(predictions)

