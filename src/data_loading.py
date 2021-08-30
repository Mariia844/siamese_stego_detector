import numpy as np
from PIL import Image

class DataLoader():
  def load(self, id: str):
    return np.load(id)


class ImageDataLoader(DataLoader):
  def __init__(self, image_size : tuple, n_channels: int):
    self.image_size = image_size
    self.n_channels = n_channels

  def load(self, id: str):
    image = Image.open(id)
    image = np.array(image.resize(self.image_size, Image.ANTIALIAS))
    image = image.astype("float32") / 255.0
    image = np.reshape(image, (*self.image_size, self.n_channels))
    return image