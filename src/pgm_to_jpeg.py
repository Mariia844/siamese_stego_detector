from data.loading import SimpleImageDataLoader
from data.writing import JpegNotCompressingImageDataWriter
import os


source = 'E:/Mary/SIAMESE_VECTOR/siamese_data/suni_0.5'
destination = 'E:/Mary/SIAMESE_VECTOR/siamese_data/suni_0.5_JPEG'
image_size = (512,512)
data_loader = SimpleImageDataLoader(image_size)
data_writer = JpegNotCompressingImageDataWriter(image_size)
for folder in os.listdir(source):
    folder_path = os.path.join(source, folder)
    for subset_folder in os.listdir(folder_path):
        subset_folder_path = os.path.join(folder_path, subset_folder)
        for image in os.listdir(subset_folder_path):
            image_path = os.path.join(subset_folder_path, image)
            print(f'Processing {image_path}')
            name, extension = os.path.splitext(image)
            target_path = os.path.join(destination, folder, subset_folder, name + '.jpeg')
            target_dir = os.path.dirname(target_path)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            data = data_loader.load(image_path)
            data_writer.write_image(data, target_path, True)

    