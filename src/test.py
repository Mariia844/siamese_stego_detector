# from common import MessageProducer, TelegramMessageProducer, get_config


# mp = MessageProducer(scope='test')

# mp.produce_message('Hello world')
# tele_config = get_config()['telegram']
# message_p = TelegramMessageProducer('ImageDataLoader', **tele_config)

# kwargs = {
#         'path': 'gegegege',
#         'error': 'asdasdasd str(e)'
#       }
# message_p.produce_message('Error while loading path: {path}, error is {error}', **kwargs)

import time
def test_read_image(imgfile, func):
    t0 = time.time()
    img = func(imgfile)
    return img, time.time() - t0

import cv2
from PIL import Image
import numpy as np
train_path = "E:/Mary/SIAMESE_JPEG/dae_data/cover/00001.jpeg"
# test_path = "E:/Mary/SIAMESE_JPEG/dae_data/stego/1"
test_image, cv2_time = test_read_image(train_path, lambda x: cv2.imread(x, cv2.IMREAD_GRAYSCALE))
# train_image = cv2.imread(test_path)
pil_image, pil_time = test_read_image(train_path, Image.open)
pil_array = np.array(pil_image)

print('CV2 shape: ', test_image.shape)
print('PIL shape: ', pil_array.shape)

print(f'Times: \n\tCV2: {cv2_time}\n\tPIL: {pil_time}')