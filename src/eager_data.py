from logging import debug
from keras.utils import Sequence, to_categorical
import numpy as np
from glob import glob
from PIL import Image
import os
from data_loading import DataLoader, ImageDataLoader

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
    self.train_paths, self.test_paths = [], []
    for i in range(index*self.batch_size, index*self.batch_size+self.batch_size):
        train_path = self.train_images_paths[self.index[i]]
        test_path = self.test_images_paths[self.index[i]]
        train_image = Image.open(train_path)
        test_image = Image.open(test_path)
        self.train_paths.append(train_path)
        self.test_paths.append(test_path)
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


class SiameseNetworkDataGenerator(Sequence):
  def __init__(self, paths_with_labels, batch_size, shuffle = True):
    self.data = []
    self.labels = []
    for value in paths_with_labels:
      key = value[0]
      images = [glob(key[0]), glob(key[1])]
      min_len = min(len(images[0]), len(images[1]))
      for i in range(min_len):
        self.data.append([[images[0][i], images[1][i]], value[1]])
        self.labels.append(value[1])
    self.data_len = len(self.data)
    self.index = [i for i in range(self.data_len)]

    self.batch_size = batch_size
    self.shuffle = shuffle
    self.on_epoch_end()
  #TODO: Implement generator and test it with siamese network
  def generator(self):
    pairs = []
    labels = []
    index = 0
    for index in range(len(self)):
      for i in range(index*self.batch_size, index*self.batch_size+self.batch_size):
        current_data = self.data[self.index[i]]
        path_data = current_data[0]
        label_data = current_data[1]
        path_1 = path_data[0]
        path_2 = path_data[1]
        image_1 = Image.open(path_1)
        image_2 = Image.open(path_2)
        image_1 = np.array(image_1.resize((512,512), Image.ANTIALIAS))
        image_2 = np.array(image_2.resize((512,512), Image.ANTIALIAS))

        image_1 = image_1.astype("float32") / 255.0
        image_2 = image_2.astype("float32") / 255.0

        image_1 = np.reshape(image_1, (512,512, 1))
        image_2 = np.reshape(image_2, (512,512, 1))
        pairs.append([image_1, image_2])
        labels.append(label_data)
    return pairs
  def __getitem__(self, index):
    pairs = []
    labels = []
    for i in range(index*self.batch_size, index*self.batch_size+self.batch_size):
      current_data = self.data[self.index[i]]
      path_data = current_data[0]
      label_data = current_data[1]
      path_1 = path_data[0]
      path_2 = path_data[1]
      image_1 = Image.open(path_1)
      image_2 = Image.open(path_2)
      image_1 = np.array(image_1.resize((512,512), Image.ANTIALIAS))
      image_2 = np.array(image_2.resize((512,512), Image.ANTIALIAS))

      image_1 = image_1.astype("float32") / 255.0
      image_2 = image_2.astype("float32") / 255.0

      image_1 = np.reshape(image_1, (512,512, 1))
      image_2 = np.reshape(image_2, (512,512, 1))
      pairs.append([image_1, image_2])
      labels.append(label_data)
    return pairs

  def __len__(self):
      return int(self.data_len / self.batch_size)

  def on_epoch_end(self):
      if (self.shuffle):
        np.random.shuffle(self.index)



class EagerDataGenerator(Sequence):
  'Generates data for Keras'
  def __init__(self, list_IDs, data_loader : DataLoader, batch_size=16, dim=(512,512), n_channels=1,
                n_classes=2, shuffle=True):
      'Initialization'
      self.dim = dim
      self.batch_size = batch_size
      self.list_IDs = list_IDs
      self.n_channels = n_channels
      self.n_classes = n_classes
      self.shuffle = shuffle
      self.id_length = len(list_IDs)
      self.batches_per_epoch = int(np.floor(self.id_length / self.batch_size))
      self.data_loader = data_loader
      self.on_epoch_end()
  def __len__(self):
      'Denotes the number of batches per epoch'
      return self.batches_per_epoch
  def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        result = self.__data_generation(list_IDs_temp)

        return result
  def on_epoch_end(self):
      'Updates indexes after each epoch'
      self.indexes = np.arange(len(self.list_IDs))
      if self.shuffle == True:
          np.random.shuffle(self.indexes)
  def __data_generation(self, list_IDs_temp):
      'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
      # Initialization
      X = np.empty((self.batch_size, *self.dim, self.n_channels))
      # y = np.empty((self.batch_size), dtype=int)

      # Generate data
      for i, ID in enumerate(list_IDs_temp):
          # Store sample
          X[i] = self.data_loader.load(ID)

          # Store class
          # y[i] = self.labels[ID]

      return X

class EagerDataGeneratorXY(EagerDataGenerator):
  def __init__(self, list_IDs, labels, data_loader : DataLoader, batch_size=16, dim=(512,512), n_channels=1,
                n_classes=2, shuffle=True):
      'Initialization'
      super(EagerDataGeneratorXY, self).__init__(list_IDs, data_loader, batch_size, dim, n_channels, n_classes, shuffle)
      self.labels = labels
  def __getitem__(self, index):
      'Generate one batch of data'
      # Generate indexes of the batch
      indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

      # Find list of IDs
      list_IDs_temp = [self.list_IDs[k] for k in indexes]

      # Generate data
      result = self.__data_generation(list_IDs_temp)

      return result
  def __data_generation(self, list_IDs_temp):
      'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
      # Initialization
      X = np.empty((self.batch_size, *self.dim, self.n_channels))
      y = np.empty((self.batch_size), dtype=int)

      # Generate data
      for i, ID in enumerate(list_IDs_temp):
          # Store sample
          X[i] = self.data_loader.load(ID)

          # Store class
          y[i] = self.labels[i]

      return X, to_categorical(y, num_classes=self.n_classes)


class SiameseStegoEagerDataGenerator(EagerDataGenerator):
  def __init__(self, images_path, label, image_size, batch_size, shuffle):
    list_IDs = glob(images_path)
    labels = None
    if isinstance(label, int):
      labels = np.full(len(list_IDs), label)
    else:
      labels = label
    super(SiameseStegoEagerDataGenerator, self).__init__(list_IDs, ImageDataLoader(image_size, 1), batch_size, image_size, 1, 2, shuffle)

class StegoData:
  def __init__(self, stego_path, stego_dae_path, cover_path, cover_dae_path, types = ["*.tif"]):
    self.stego_path = stego_path
    self.stego_dae_path = stego_dae_path
    self.cover_path = cover_path
    self.cover_dae_path = cover_dae_path
    self.types = types

    self.stego_paths = self.__get_paths(self.stego_path)
    self.stego_dae_paths = self.__get_paths(self.stego_dae_path)
    self.cover_paths = self.__get_paths(self.cover_path)
    self.cover_dae_paths = self.__get_paths(self.cover_dae_path)

    self.min_index = min(len(self.stego_paths), len(self.stego_dae_paths), len(self.cover_paths), len(self.cover_dae_paths))

    # 0 
    self.cover_paths = [self.cover_paths[:self.min_index], self.cover_dae_paths[:self.min_index]]
    # 1
    self.stego_paths = [self.stego_paths[:self.min_index], self.stego_dae_paths[:self.min_index]]

  def __get_paths(self, path):
    result = []
    for type in self.types:
      temp_result = glob(os.path.join(path, type))
      if (len(temp_result) == 0):
        raise AssertionError(f'Path {os.path.join(path, type)} is empty!')
      temp_result.pop()
      result.extend(temp_result)
    return result
  
class TwoLegStegoDataGenerator(Sequence):
  def __init__(self, stego_data : StegoData, data_loader : DataLoader, batch_size, shuffle = True):
      self.stego_data = stego_data
      self.data_loader = data_loader
      self.batch_size = batch_size
      self.shuffle = shuffle
      self.on_epoch_end()

  def __getitem__(self, index):
      'Generate one batch of data'
      # Generate indexes of the batch
      indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

      # Find list of IDs
      if index == 235:
        ax = 2
      result = [[], []]
      labels = []
      for idx in indexes:
        label = 0
        if idx >= (self.stego_data.min_index) - 1:
          label = 1
        # data_index = 
        path1, path2 = None, None
        if label == 0:
          path1, path2 = self.stego_data.cover_paths[0][idx], self.stego_data.cover_paths[1][idx]
        else:
          path1, path2 = self.stego_data.stego_paths[0][idx - self.stego_data.min_index], self.stego_data.stego_paths[1][idx - self.stego_data.min_index]
        result[0].append(self.data_loader.load(path1))
        result[1].append(self.data_loader.load(path2))
        labels.append([0,1] if label == 0 else [1,0])

      # Generate data
      # result[0] = np.array(result[0])
      # result[1] = np.array(result[1])
      result = ([np.array(x) for x in result], np.array(labels).astype('float32'))
      if (result[0][0].shape == (0,)):
        print('\n\nERROR SHAPE.\n\tRequested index: ', index, '\n\tLabels: ', labels, '\n\n\n')
      return result
  def on_epoch_end(self):
      'Updates indexes after each epoch'
      self.indexes = np.arange(self.stego_data.min_index*2)
      if self.shuffle == True:
          np.random.shuffle(self.indexes)
  def __len__(self):
    """Number of batch in the Sequence.

    Returns:
        The number of batches in the Sequence.
    """
    return int(np.floor(self.stego_data.min_index * 2 / self.batch_size))