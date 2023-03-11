import torch
from torch_geometric.data import Dataset

class MyOwnDataset(Dataset):
    def __init__(self, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)