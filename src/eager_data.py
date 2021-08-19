from keras.utils import Sequence
import numpy as np
from glob import glob
from PIL import Image
class DataGenerator(Sequence):

  def __init__(self, train_path, test_path, batch_size=32, shuffle=True, start_index = 0, take = -1):
  
    self.train_images_paths = glob(train_path)
    self.test_images_paths = glob(test_path)
    if (start_index != 0):
        self.train_images_paths = self.train_images_paths[start_index:]
        self.test_images_paths = self.test_images_paths[start_index:]
    if (take != -1):
        self.train_images_paths = self.train_images_paths[:take]
        self.test_images_paths = self.test_images_paths[:take]
    self.train_length = len(self.train_images_paths)
    self.test_length = len(self.test_images_paths)
    if (self.train_length != self.test_length):
        raise AssertionError(f'train and test lengths should be the same ({self.train_length} != {self.test_length})')
    self.index = [i for i in range(len(self.train_images_paths))]
    
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.on_epoch_end()

  def __len__(self):
    return int(self.train_length / self.batch_size)

  def __getitem__(self, index):
    train, test = [], []
    for i in range(index, index+self.batch_size):
        train_path = self.train_images_paths[self.index[i]]
        test_path = self.test_images_paths[self.index[i]]
        train_image = Image.open(train_path)
        test_image = Image.open(test_path)

        train_image = np.array(train_image.resize((512,512), Image.ANTIALIAS))
        test_image = np.array(test_image.resize((512,512), Image.ANTIALIAS))

        train_image = train_image.astype("float32") / 255.0
        test_image = test_image.astype("float32") / 255.0

        train_image = np.reshape(train_image, (512,512, 1))
        test_image = np.reshape(test_image, (512,512, 1))

        train.append(train_image)
        test.append(test_image)

    return np.array(train), np.array(test)

  def on_epoch_end(self):
      if (self.shuffle):
          np.random.shuffle(self.index)