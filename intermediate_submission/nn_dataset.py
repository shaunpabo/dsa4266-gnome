import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd

class GeneDataset(Dataset):
    def __init__(self, X_df, y_df):
        self.X_df = X_df
        self.y_df = y_df

    def __len__(self):
        return len(self.X_df)

    def __getitem__(self, idx):
        X_data = self.X_df[idx]
        y_data = self.y_df.iloc[idx]
        return X_data, y_data