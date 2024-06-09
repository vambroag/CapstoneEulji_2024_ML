import config
import pandas as pd

def load_dataset(file_name):
    _data = pd.read_csv(config.DataFramePath, sep='\t', usecols=config.columns)
    return _data