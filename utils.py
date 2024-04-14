import torch
from data import TrainDataset, ValidDataset, TestDataset

def make_train_data_loader(cfg):
    train_datasets = TrainDataset(**cfg.path.train)
    train_dataloader = torch.utils.data.DataLoader(
        train_datasets, shuffle=True, **cfg.loader
    )

    valid_datasets = ValidDataset(**cfg.path.valid)
    valid_dataloader = torch.utils.data.DataLoader(
        valid_datasets, shuffle=False, **cfg.loader
    )

    return train_dataloader, valid_dataloader

def make_test_data_loader(cfg):
    test_datasets = TestDataset(**cfg.path.test)
    test_dataloader = torch.utils.data.DataLoader(
        test_datasets, shuffle=False, **cfg.loader
    )

    return test_dataloader
