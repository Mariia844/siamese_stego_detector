import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, MaxPool2D, Flatten, Lambda, Dense, Dropout

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


def get_siamese_network(input_shape : tuple = (512,512,1), dae_input: Model = None) -> Model:
    input = dae_input
    if input == None:
        input = Input(input_shape)
    x = Conv2D(16, (7, 7), activation="tanh", strides=(2,2))(input)
    x = BatchNormalization()(x)
    x = Conv2D(32, (5, 5), activation="tanh")(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2,1), strides=(2,2))(x)
    x = Conv2D(64, (3, 3), activation="tanh")(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2,1), strides=(2,2))(x)
    x = Conv2D(16, (1, 1), activation="tanh")(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    embedding_network = Model(input, x)


    input_1 = Input((512,512,1))
    input_2 = Input((512,512,1))

    tower_1 = embedding_network(input_1)
    tower_2 = embedding_network(input_2)

    x = Lambda(l1_distance_v2)([tower_1, tower_2])
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.2)(x)
    output_layer = Dense(2, activation="sigmoid")(x)

    siamese = Model(inputs=[input_1, input_2], outputs=output_layer)
    return siamese