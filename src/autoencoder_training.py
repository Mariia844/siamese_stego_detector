from numpy.lib.npyio import save
from tensorflow.python.keras.callbacks import ModelCheckpoint

from eager_data import DataGenerator

import ml_utils
import os
from datetime import datetime
import common
from autoencoder.model import get_compiled_model, get_compiled_model_with_weights

import win32file
win32file._setmaxstdio(2048)

def main():
    config = common.get_config()
    images_config = config['images']
    count = int(images_config['count'])
    size = int(images_config['image_size'])
    target_size = (size, size)
    stego_path = images_config['stego_path']
    cover_path = images_config['cover_path']
    save_path = images_config['save_path']
    create_dir = common.str2bool(images_config['create_datetime_dir'])
    load_model = common.str2bool(images_config['load_model'])
    epochs = int(images_config['epochs'])
    batch_size = int(images_config['batch_size'])
    model_path = images_config['model_path']
    if (create_dir):
        now = datetime.now()
        d1 = now.strftime("%d_%m_%Y_%H_%M_%S")
        save_path = os.path.join(save_path, d1)
        os.makedirs(save_path)
    train_generator = DataGenerator(train_path=stego_path, test_path=cover_path, batch_size=16, shuffle=False, take = count)
    validation_generator = DataGenerator(train_path=stego_path, test_path=cover_path, batch_size=16, shuffle=False, start_index= count, take = count)
    input_shape = (*target_size, 1)

    autoencoder = None

    if load_model:
        autoencoder = get_compiled_model_with_weights(model_path, input_shape)
    else:
        autoencoder = get_compiled_model(input_shape)
    filepath = save_path + "/saved-model-ep_{epoch:02d}-loss_{loss:.5f}.hdf5"
    
    checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1,
        save_best_only=False, mode='auto', period=1)
    
    history = autoencoder.fit(
        x = train_generator,
        epochs = epochs,
        batch_size = batch_size,
        shuffle = False,    
        validation_data = validation_generator,
        callbacks=[checkpoint])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    ml_utils.save_model_history_csv(history, os.path.join(save_path, 'history.csv'))
    common.send_message(text=f'DAE training completed')
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        common.send_message(e)