import glob
import os
from threading import current_thread

from numpy.lib.utils import source
from autoencoder_training import get_model
from common import get_config
from tensorflow.keras import layers, metrics
import telebot
from PIL import Image
import numpy as np
from eager_data import DataGenerator

def read_image(path):
    train_image = Image.open(path)
    train_image = np.array(train_image.resize((512,512), Image.ANTIALIAS))
    train_image = train_image.astype("float32") / 255.0
    train_image = np.reshape(train_image, (512,512, 1))
    return train_image
def write_image(data, path, overwrite = False):
    if not overwrite and os.path.exists(path):
        return 
    image_arr = np.reshape(data, (512,512))
    image = Image.fromarray(image_arr)
    try:
        image.save(path)
    except OSError as e:
        print(e)
        import imageio
        imageio.imwrite(path, image_arr)
def has_elements(iter):
  from itertools import tee
  iter, any_check = tee(iter)
  try:
    any_check.next()
    return True, iter
  except StopIteration:
    return False, iter
config = get_config()
images_config = config['images']
telegram_config = config['telegram']
task_config = config['DAE_task']

bot = telebot.TeleBot(telegram_config['token'])
count = int(images_config['count'])
size = int(images_config['image_size'])
cover_path = images_config['cover_path']
target_size = (size, size)
chat_id = telegram_config['chat_id']
input_shape = (*target_size, 1)

autoencoder = get_model(input_shape)
autoencoder.compile(optimizer="adadelta", loss="binary_crossentropy", metrics=[
                    metrics.MeanSquaredError()])

task_path = task_config['path']
target_path = task_config['target_path']
weights_path = task_config['weights_path']

autoencoder.load_weights(weights_path)

def encode_and_write_images(source_folder, target_folder):
    extensions = set(os.path.splitext(f)[1] for dir,dirs,files in os.walk(source_folder) for f in files)
    if (len(extensions) != 1):
        print('Folders should contain only one extension, but it contains ', len(extensions), '; they are ', extensions)
        return
    extension = extensions.pop()
    stego_path = f"{source_folder}\\*{extension}"
    data_generator = DataGenerator(train_path=stego_path, test_path=cover_path, batch_size=16, shuffle=False, take = count)
    image_iterator = iter(data_generator)
    
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    k = 0
    batch_index = 1
    while True:
        print('Batch ', batch_index)
        batch_index += 1
        next_image_batch = next(image_iterator, None)
        if next_image_batch is None:
            break
        need_to_predict = False
        for j in range(data_generator.batch_size):
            path = data_generator.train_images_paths[k+j]
            folder, filename = os.path.split(path)
            target_train_path =  os.path.join(target_folder, filename)
            if not os.path.exists(target_train_path):
                need_to_predict = True
                break
        k = k + data_generator.batch_size

        if need_to_predict:
            prediction_images = autoencoder.predict(next_image_batch)
            for j in range(data_generator.batch_size):
                path = data_generator.train_paths[j]
                corresponding_image = prediction_images[j]
                folder, filename = os.path.split(path)
                target_filename = os.path.join(target_folder, filename)
                write_image(corresponding_image, target_filename)

if __name__ == "__main__":
    folder, filename = os.path.split(cover_path)
    encode_and_write_images(folder, os.path.join(target_path, 'cover'))
    algorithms_names = os.listdir(task_path)
    levels_names = [os.listdir(os.path.join(task_path, name)) for name in algorithms_names]
    # print(algorithms_names)
    # print(levels_names)
    i = 0
    for name in algorithms_names:
        for level in levels_names[i]:
            current_path = os.path.join(task_path, name, level)
            current_target_path = os.path.join(target_path, name, level)
            encode_and_write_images(current_path, current_target_path)   
        i += 1


