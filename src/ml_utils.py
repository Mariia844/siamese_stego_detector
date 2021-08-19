import pandas as pd


def save_model_history_csv(history, file_path = 'history.csv'):
    hist_df = pd.DataFrame(history.history) 
    with open(file_path, mode='w') as f:
        hist_df.to_csv(f)

