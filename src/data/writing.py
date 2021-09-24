from PIL import Image
import imageio
import os
import numpy as np
import logging
import shutil
from skimage import img_as_ubyte
from tensorflow.python.training.tracking.base import TrackableReference


class ImageWriter():
    def __init__(self, target_size: tuple = None, convert_to = None):
        self.target_size = target_size
        self.convert_to = convert_to
    def write_image(self, data, target_path, overwrite = False):
        if not overwrite and os.path.exists(target_path):
            return         
        imageio.imwrite(target_path, img_as_ubyte(np.clip(data.reshape(self.target_size), -1, 1)))

class JpegNotCompressingImageDataWriter(ImageWriter):
    def __init__(self, target_size: tuple):
        super().__init__(target_size=target_size)
    def write_image(self, data : np.ndarray, target_path : str, overwrite=False):
        if not overwrite and os.path.exists(target_path):
            return
        data = data.reshape(self.target_size)
        img = Image.fromarray(data)
        img.save(target_path, format='JPEG', quality=100, subsampling=0)

class CopyFileWriter(ImageWriter):
    def __init__(self):
        super().__init__()
    def write_image(self, data, target_path, overwrite = False):
        if not overwrite and os.path.exists(target_path):
            return
        shutil.copyfile(data, target_path)


class TrainTestSplitWriter():
    def __init__(self, 
        target_path : str,
        image_writer : ImageWriter,
        total_count: int,
        train_count : int = None, 
        train_part: float = None) -> None:

        assert train_count == None or train_part == None, "train_count can not be specified with train_part"
        assert train_count == None or train_count > 0, "Train count should be positive number"
        assert train_count == None or train_count <= total_count, "Train count should less or equal to total count"
        assert train_part == None or train_part > 0 and train_part < 1, "Train part should be positive float between 0 and 1"
        # assert batch_size > 0, "Batch size should be positive number"
        assert total_count > 0, "Total count should be positive number"
        # assert total_count % batch_size == 0, "Total count should contain integer number of batches"


        self.target_path = target_path
        self.train_count = train_count
        self.train_part = train_part

        self.current_part = 'training'
        if train_count is not None:
            self.switch_index = train_count
        else:
            self.switch_index = int(total_count * self.train_part)
        self.current_index = -1

        self.check_switch()
        self.check_folder()
        self.image_writer = image_writer

    def check_folder(self):
        self.save_path = os.path.join(self.target_path, self.current_part)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
    def check_switch(self):
        if (self.current_index == self.switch_index):
            logging.debug(f'Switched to validation on index {self.current_index}')
            self.current_part = 'validation'
            self.check_folder()
            
    def write_next_item(self, data, file_name):
        self.current_index += 1
        self.check_switch()
        save_current_image_path = os.path.join(self.save_path, file_name)
        self.image_writer.write_image(data, save_current_image_path)
        

