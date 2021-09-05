import pandas as pd
import pickle
def dump_object(obj, filename):
    with open(filename, 'w') as fh:
        pickle.dump(obj, fh)

def save_model_history_csv(history, file_path = 'history.csv'):
    hist_df = pd.DataFrame(history.history) 
    with open(file_path, mode='w') as f:
        hist_df.to_csv(f)

