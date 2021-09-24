import os

from autoencoder.model import get_compiled_model_with_weights
from common import get_config, str2bool
from PIL import Image
import numpy as np
from eager_data import DataGenerator
from data_loading import OpenCVImageDataLoader

data_loader = OpenCVImageDataLoader((512,512), 1)

from data.writing import ImageWriter, TrainTestSplitWriter

def read_image(path):
    # train_image = Image.open(path)
    # train_image = np.array(train_image.resize((512,512), Image.ANTIALIAS))
    # train_image = train_image.astype("float32") / 255.0
    # train_image = np.reshape(train_image, (512,512, 1))
    return data_loader.load(path)
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
count = int(images_config['count'])
size = int(images_config['image_size'])
cover_path = images_config['cover_path']
target_size = (size, size)
chat_id = telegram_config['chat_id']
input_shape = (*target_size, 1)
batch_size = int(images_config['batch_size'])

task_path = task_config['path']
target_path = task_config['target_path']
weights_path = task_config['weights_path']
train_part = float(task_config['train_part'])
single_level = str2bool(task_config['single_level'])

autoencoder = get_compiled_model_with_weights(weights_path, input_shape)


def encode_and_write_images(source_folder, target_folder):
    files = [f for dir,dirs,files in os.walk(source_folder) for f in files]
    files = files[:count]
    extensions = [os.path.splitext(f)[1] for f in files]
    extensions = set(extensions)
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
    image_writer = ImageWriter(target_size, convert_to=None)
    split_writer = TrainTestSplitWriter(target_folder, image_writer, len(files), train_part=train_part)
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
                # target_filename = os.path.join(target_folder, filename)
                # write_image(corresponding_image, target_filename)
                split_writer.write_next_item(corresponding_image, filename)

if __name__ == "__main__":
    folder, filename = os.path.split(cover_path)
    # encode_and_write_images(folder, os.path.join(target_path, 'dae_cover'))
    if single_level:
        encode_and_write_images(task_path, os.path.join(target_path, 'dae_stego'))
    else:
        algorithms_names = os.listdir(task_path)
        levels_names = [os.listdir(os.path.join(task_path, name)) for name in algorithms_names]
        i = 0

        for name in algorithms_names:
            for level in levels_names[i]:
                current_path = os.path.join(task_path, name, level)
                current_target_path = os.path.join(target_path, name, level)
                encode_and_write_images(current_path, current_target_path)   
            i += 1

    


