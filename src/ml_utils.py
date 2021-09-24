import pandas as pd
import pickle
import os
def dump_object(obj, filename):
    with open(filename, 'w') as fh:
        pickle.dump(obj, fh)

def save_model_history_csv(history, file_path = 'history.csv'):
    hist_df = pd.DataFrame(history.history) 
    with open(file_path, mode='w') as f:
        hist_df.to_csv(f)


import tensorflow.keras as keras
class PrintCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))
class SaveStatsCallback(keras.callbacks.Callback):
    def __init__(self, history_path):
        self.history_path = history_path
        self.folder = os.path.dirname(history_path)
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        super().__init__()
    def on_epoch_end(self, epoch, logs=None):
        
        mode = 'w'
        header = True
        logs['epoch'] = epoch
        df = pd.DataFrame([logs])

        if (os.path.exists(self.history_path)):
            mode = 'a'
            header = False

        with open(self.history_path, mode) as f:
            df.to_csv(f, header=header)
        # keys = list(logs.keys())
        # print("End epoch {} of training; got log keys: {}".format(epoch, keys))