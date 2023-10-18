import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd

class GeneDataset(Dataset):
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath, index_col=0)
        self.X_df = self.df.drop(columns=["label", "seq"])
        self.y_df = self.df.label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        X_data = self.X_df.iloc[idx, :9]
        y_data = self.y_df.iloc[idx]
        return X_data.values, y_data