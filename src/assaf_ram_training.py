from autoencoder.model import get_autoencoder
from tensorflow.keras import metrics
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, MaxPool2D, Flatten, Lambda, Dense, Dropout

from data_loading import ImageDataLoader
from eager_data import EagerDataGenerator

import glob
import os
from PIL import Image
from random import shuffle
import joblib
import numpy as np 
from time import time
from sklearn.model_selection import train_test_split
import gc
DAE_WEIGHTS_PATH = "E:/Mary/SIAMESE_VECTOR/DAE_training_suni_0.4/20_09_2021_19_35_36/saved-model-ep_300-loss_0.59648.hdf5"
#IMAGES_PATH = "E:/Mary/SIAMESE_VECTOR/siamese_data/suni_0.5/cover_1k/training/*.pgm"
PATH = "E:/ml/notebook_train/"
EMBEDDING_ALGORYTHM='MiPOD';
DATASET = 'ALASKA'
DATA_PATH = os.path.join(PATH, 'dataset', DATASET, EMBEDDING_ALGORYTHM)
ASSAF_SAVE_PATH = os.path.join(PATH, 'assaf')
#Embedding level - change it
LEVEL = 40

ALGORITHMS = [str(LEVEL)]
IMG_SIZE = 512
RGB = False
DROPOUT_RATE=0.1
EPOCHS = 100
BATCH_SIZE = 4
IMAGES_TO_PICK = 6668
CHECKPOINT_PATH = ASSAF_SAVE_PATH + f"/checkpoints/training_{DATASET}_{EMBEDDING_ALGORYTHM}_{LEVEL}/cp-epoch-"+ "{epoch:04d}.ckpt"
SAVE_WEIGTHS_PATH = ASSAF_SAVE_PATH + f"/weights/training_{DATASET}_{EMBEDDING_ALGORYTHM}_{LEVEL}/model_epoch-"+ "{epoch:04d}_loss-{loss:04f}_val_loss-{val_loss:04f}_auc-{auc:04f}_val_auc-{val_auc:04f}.hdf5"

# FILES = glob.glob(IMAGES_PATH)

dae_model = get_autoencoder()
dae_model.trainable = False
dae_model.load_weights(DAE_WEIGHTS_PATH)

def load_image(data):
    i, j, img_path, labels = data
    
    img = Image.open(img_path)
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
    label = labels[i][j]
    return [np.array(img), label]

def load_training_data_multi(n_images=100, algorithms = ALGORITHMS, data_path = DATA_PATH):
    train_data = []
    data_paths = [os.listdir(os.path.join(DATA_PATH, alg)) for alg in ['Cover'] + algorithms]
    labels = [np.zeros(n_images)]
    for _ in range(len(algorithms)):
        labels.append(np.ones(n_images))
    print('Loading...')
    for i, image_path in enumerate(data_paths):
        print(f'\t {i+1}-th folder')
        
        train_data_alg = joblib.Parallel(n_jobs=4, backend='threading')(
            joblib.delayed(load_image)([i, j, os.path.join(DATA_PATH, [['Cover'] + algorithms][0][i], img_p), labels]) for j, img_p in enumerate(image_path[:n_images]))

        train_data.extend(train_data_alg)
        
    shuffle(train_data)
    return train_data
heavy_memory_storage = []
def clear_heavy_memory_storage():
    global heavy_memory_storage
    while (len(heavy_memory_storage) > 0):
        el = heavy_memory_storage.pop()
        del el
    gc.collect()
def get_train_data():
    global heavy_memory_storage
    training_data = load_training_data_multi(n_images=IMAGES_TO_PICK)
    channels_count_in_image = 3 if RGB else 1
    trainImages = np.array([i[0] for i in training_data]).reshape(-1, IMG_SIZE, IMG_SIZE, channels_count_in_image)
    trainLabels = np.array([i[1] for i in training_data], dtype=float)
    heavy_memory_storage.append(training_data)
    heavy_memory_storage.append(trainLabels)
    heavy_memory_storage.append(trainImages)
    return train_test_split(trainImages, trainLabels, random_state=42, stratify=trainLabels)
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


def get_siamese_network(input_shape : tuple = (IMG_SIZE,IMG_SIZE,1)) -> Model:
    
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
    output_layer = Dense(1, activation="sigmoid")(x)

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

checkpoint = tf.keras.callbacks.ModelCheckpoint(SAVE_WEIGTHS_PATH, monitor='accuracy', verbose=1,
    save_best_only=False, mode='auto', period=1)
folder, tail = os.path.split(SAVE_WEIGTHS_PATH)
if not os.path.exists(folder):
    os.makedirs(folder)

X_train, X_val, y_train, y_val = get_train_data()
clear_heavy_memory_storage()

history = siamese.fit(X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS, 
        callbacks=[checkpoint],
        verbose=1)
file_path = os.path.join(ASSAF_SAVE_PATH, f'training_{}{EMBEDDING_ALGORYTHM}_{LEVEL}_history.csv')
import pandas as pd
hist_df = pd.DataFrame(history.history) 
import win32file
win32file._setmaxstdio(2048)
with open(file_path, mode='w') as f:
    hist_df.to_csv(f)

# generator = EagerDataGenerator(FILES, ImageDataLoader(IMAGE_SIZE, 1))
# next_batch = next(iter(generator))

# predictions = siamese.predict(next_batch)
# print(predictions)

