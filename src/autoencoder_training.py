import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras.models import Model

from PIL import Image
import configparser
import glob
from sklearn.model_selection import train_test_split

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

def get_images(path, images_count, target_size = (512,512)):
    images = glob.glob(path)[:images_count]
    return [load_image(img, target_size) for img in images]

def load_image(path, target_size):
    img = Image.open(path)
    img = img.resize(target_size, Image.ANTIALIAS)
    arr = np.array(img)
    arr = arr.astype("float32") / 255.0
    return np.reshape(arr, (*target_size, 1))
    #return [np.expand_dims(np.array(img), axis=2)]
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
    count = int(config['images']['count'])
    size = int(config['images']['image_size'])
    target_size = (size, size)
    cover_train, cover_test = train_test_split(get_images(config['images']['stego_path'], count, target_size))
    cover_train, cover_test = np.array(cover_train), np.array(cover_test)
    input_shape = (*target_size, 1)

    autoencoder = get_model(input_shape)
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
    # display(cover_test, cover_test)
    autoencoder.fit(
        x = cover_train, 
        y = cover_train,
        epochs = 70,
        batch_size = 16,
        shuffle = True, 
        validation_data = (cover_test, cover_test))
    predictions = autoencoder.predict(cover_test)
    # display(cover_test, predictions)