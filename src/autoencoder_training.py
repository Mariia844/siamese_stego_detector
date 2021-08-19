import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers, metrics
from tensorflow.keras.models import Model

import configparser

from tensorflow.python.keras.callbacks import ModelCheckpoint

from eager_data import DataGenerator

import ml_utils
import os
from datetime import datetime
import telebot

def get_model(input_shape = (512,512,1)):
    i = layers.Input(shape=input_shape)

    x = layers.Conv2D(4, (7, 7))(i)
    x = layers.Conv2D(10, (5,5))(x)
    x = layers.Conv2D(20, (3,3))(x)
    x = layers.Conv2DTranspose(20, (3,3))(x)
    x = layers.Conv2DTranspose(10, (5,5))(x)
    x = layers.Conv2DTranspose(4, (7,7))(x)
    x = layers.Conv2DTranspose(1, (1,1))(x)
    return Model(i, x)

def get_config():
    config = configparser.ConfigParser()
    #config.read('../../config.ini')
    config.read('../config.ini')
    return config

def display(array1, array2):
    """
    Displays ten random images from each one of the supplied arrays.
    """

    n = 10

    indices = np.random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]

    plt.figure(figsize=(20, 4))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1.reshape(512, 512))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2.reshape(512, 512))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()
if __name__ == "__main__":
    
    config = get_config()
    images_config = config['images']
    telegram_config = config['telegram']
    bot = telebot.TeleBot(telegram_config['token'])
    count = int(images_config['count'])
    size = int(images_config['image_size'])
    target_size = (size, size)
    stego_path = images_config['stego_path']
    cover_path = images_config['cover_path']
    save_path = images_config['save_path']
    create_dir = bool(images_config['create_datetime_dir'])
    if (create_dir):
        now = datetime.now()
        d1 = now.strftime("%d_%m_%Y_%H_%M_%S")
        save_path = os.path.join(save_path, d1)
        os.makedirs(save_path)
    train_generator = DataGenerator(train_path=stego_path, test_path=cover_path, batch_size=16, shuffle=False, take = count)
    validation_generator = DataGenerator(train_path=stego_path, test_path=cover_path, batch_size=16, shuffle=False, start_index= count, take = count)
    input_shape = (*target_size, 1)

    autoencoder = get_model(input_shape)
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy", metrics=[
                        metrics.MeanSquaredError()])
    filepath = save_path + "/saved-model-ep_{epoch:02d}-loss_{loss:.2f}-val_loss_{val_loss:.2f}.hdf5"
    
    checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1,
        save_best_only=False, mode='auto', period=1)
    
    history = autoencoder.fit(
        x = train_generator,
        epochs = 200,
        batch_size = 16,
        shuffle = False, 
        validation_data = validation_generator,
        callbacks=[checkpoint])
    ml_utils.save_model_history_csv(history, os.path.join(save_path, 'history.csv'))
    bot.send_message(chat_id=telegram_config['chat_id'], text=f'DAE training completed')