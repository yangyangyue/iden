"""
MIT License
Copyright (c) 2024-present lily.

written by lily
email: lily231147@gmail.com
"""

import argparse
import random

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import torch
from torch.utils.data import ConcatDataset, random_split, Subset

from dataset import *
from lightning_module import NilmDataModule, NilmNet

def train_once(method, fit_dataset, save_name):
    # 预训练
    train_set, val_set = random_split(fit_dataset, [0.8, 0.2])
    datamodule = NilmDataModule(train_set=train_set, val_set=val_set, bs=256)
    model = NilmNet(method)
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints/', filename=save_name, monitor="f1")
    early_stop_callback = EarlyStopping(monitor="f1", patience=20)
    trainer = pl.Trainer(devices="auto", accelerator="auto", max_epochs=80, callbacks=[checkpoint_callback, early_stop_callback], log_every_n_steps=10)
    trainer.fit(model, datamodule=datamodule)

def train(method, fithouses, multiple):
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('high')
    # 数据
    sets = get_sets(fithouses, 'fit')
    if multiple:
        # 多设备直接训练
        train_once(method, ConcatDataset(fit_datasets), f'{method}-{fithouses}-kmdwf')
    else:
        # 单设备依次训练
        for i, app_abb in enumerate("kmdwf"):
            print(f"train {app_abb} ...", flush=True)
            train_once(method, sets[i], f'{method}-{fithouses}-{app_abb}')

if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='mada') # 方法 mada vae s2p s2s
    parser.add_argument('--fithouses', type=str, default='ukdale15') # 指定用于训练的房屋 ukdale15
    parser.add_argument('--multiple', action='store_true') # 指定则是多设备
    args = parser.parse_args()
    # train
    train(args.method, args.fithouses, args.multiple)
    