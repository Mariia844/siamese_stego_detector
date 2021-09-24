import numpy as np
from PIL import Image
import cv2



class DataLoader():
  def load(self, id: str):
    return np.load(id)



class ImageDataLoader(DataLoader):
  def __init__(self, image_size : tuple, n_channels: int):
    self.image_size = image_size
    self.n_channels = n_channels

  def load(self, id: str):
    image = Image.open(id)
    result = image.copy()
    image.close()
    np_result = np.array(result.resize(self.image_size, Image.ANTIALIAS))
    np_result = np_result.astype("float32") / 255.0
    np_result = np.reshape(np_result, (*self.image_size, self.n_channels))
    return np_result
class OpenCVImageDataLoader(DataLoader):
  def __init__(self, image_size : tuple, n_channels: int):
    self.image_size = image_size
    self.n_channels = n_channels
    self.__cv2_flag = cv2.IMREAD_GRAYSCALE if self.n_channels == 1 else cv2.IMREAD_UNCHANGED
  def load(self, id: str):
    image = cv2.imread(id, self.__cv2_flag)
    image = image.astype("float32") / 255.0
    image = np.reshape(image, (*self.image_size, self.n_channels))
    return image