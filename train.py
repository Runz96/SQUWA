# Standard library imports
from datetime import datetime
import os

# Third-party imports
import argparse
import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Local application/library specific imports
from models.model_adapt import ModelModule
from utils import make_train_data_loader


@hydra.main(version_base=None, config_path="config", config_name="train_adapt")
def main(cfg: DictConfig) -> None:

    print("Data loading", flush=True)
    train_dataloader, valid_dataloader = make_train_data_loader(
        cfg.data
    )
    print("Data loaded", flush=True)

    total_training_steps = len(train_dataloader) * cfg.trainer.parameters.max_epochs

    model = ModelModule(**cfg.model, total_training_steps=total_training_steps)
    
    ckpt_callback = ModelCheckpoint(**cfg.trainer.callbacks.model_checkpoint)
    early_stop_callback = EarlyStopping(**cfg.trainer.callbacks.early_stop)
    callbacks = [ckpt_callback, early_stop_callback]

    logger = TensorBoardLogger(**cfg.trainer.logger.tensorboard)

    trainer = pl.Trainer(**cfg.trainer.parameters, callbacks=callbacks, logger=logger)

    ckpt_path = None
    if cfg.experiment.resume_ckpt:
        ckpt_path = cfg.experiment.ckpt_path

    if cfg.experiment.train:
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=valid_dataloader,
            ckpt_path=ckpt_path,
        )

if __name__ == "__main__":
    main()
    print("Done!")
