import numpy as np
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, x_signal_path: str, x_quality_path: str, x_cluster_path: str, ypath: str):
        super(TrainDataset, self).__init__()

        # Loading entire datasets into RAM
        self.x = np.load(x_signal_path)
        self.x_quality = np.load(x_quality_path)
        self.clusters = np.load(x_cluster_path).astype(np.int64)
        self.y = np.load(ypath).astype(np.int64)

    def __getitem__(self, index):
        x_get = self.x[index]
        x_quality_get = self.x_quality[index]
        cluster_get = self.clusters[index]
        y_get = self.y[index]
        return x_get, x_quality_get, cluster_get, y_get

    def __len__(self):
        return len(self.y)
    
class ValidDataset(Dataset):
    def __init__(self, x_signal_path: str, x_quality_path: str, ypath: str):
        super(ValidDataset, self).__init__()

        # Loading entire datasets into RAM
        self.x = np.load(x_signal_path)
        self.x_quality = np.load(x_quality_path)
        self.y = np.load(ypath).astype(np.int64)

    def __getitem__(self, index):
        x_get = self.x[index]
        x_quality_get = self.x_quality[index]
        y_get = self.y[index]
        return x_get, x_quality_get, y_get

    def __len__(self):
        return len(self.y)
    
class TestDataset(Dataset):
    def __init__(self, x_signal_path: str, x_quality_path: str, ypath: str):
        super(TestDataset, self).__init__()

        # Loading entire datasets into RAM
        self.x = np.load(x_signal_path)
        self.x_quality = np.load(x_quality_path)
        self.y = np.load(ypath).astype(np.int64)

    def __getitem__(self, index):
        x_get = self.x[index]
        x_quality_get = self.x_quality[index]
        y_get = self.y[index]
        return x_get, x_quality_get, y_get

    def __len__(self):
        return len(self.y)
