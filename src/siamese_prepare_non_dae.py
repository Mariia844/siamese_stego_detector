from data.writing import CopyFileWriter, TrainTestSplitWriter
from common import get_config, str2bool
import os 
from glob import glob

config = get_config()
images_config = config['images']

task_config = config['write_task']
count = int(images_config['count'])
size = int(images_config['image_size'])
cover_path = images_config['cover_path']
target_size = (size, size)

input_shape = (*target_size, 1)
batch_size = int(images_config['batch_size'])

task_path = task_config['path']
target_path = task_config['target_path']
train_part = float(task_config['train_part'])
single_level = str2bool(task_config['single_level'])


writer = CopyFileWriter()
def write_images(pattern, target_folder):
    files = glob(pattern)[:count]
    split_writer = TrainTestSplitWriter(target_path=target_folder, image_writer=writer, total_count=len(files), train_part=train_part)
    for file in files:
        filename = os.path.split(file)[1]
        split_writer.write_next_item(file, filename)
        


if __name__ == "__main__":
    folder, filename = os.path.split(cover_path)
    write_images(cover_path, os.path.join(target_path, 'cover'))
    if single_level:
        write_images(task_path, os.path.join(target_path, 'stego'))
    else:
        algorithms_names = os.listdir(task_path)
        levels_names = [os.listdir(os.path.join(task_path, name)) for name in algorithms_names]
        i = 0

        for name in algorithms_names:
            for level in levels_names[i]:
                current_path = os.path.join(task_path, name, level)
                current_target_path = os.path.join(target_path, name, level)
                write_images(current_path, current_target_path)   
            i += 1
