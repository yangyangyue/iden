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

import matplotlib.pyplot as plt


def train(method, fithouses):
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('high')
    # 数据
    sets = get_sets(fithouses, 'fit')
    # k_set=sets[0]
    # k=0
    # for win in k_set:
    #     ids, stamps, aggs, poses, clzes = win
    #     if len(poses) == 0:
    #         continue
    #     k+=1
    #     if k>10:
    #         break
    #     plt.figure()
    #     plt.plot(aggs)
    #     for pos in poses:
    #         plt.axvline(pos, color='r')
    #     plt.savefig(f'{k}.png', dpi=600, bbox_inches='tight')
    #     plt.close()
    for app_abb, app_set in zip("kmdwf", sets):
        print(f"train {app_abb} ...", flush=True)
        train_set, val_set = random_split(app_set, [0.8, 0.2])
        datamodule = NilmDataModule(train_set=train_set, val_set=val_set, bs=256)
        model = NilmNet(method)
        checkpoint_callback = ModelCheckpoint(dirpath='checkpoints/', filename=f'{method}-{fithouses}-{app_abb}', monitor="f1")
        early_stop_callback = EarlyStopping(monitor="f1", mode="max", patience=20)
        trainer = pl.Trainer(devices="auto", accelerator="auto", max_epochs=80, callbacks=[checkpoint_callback, early_stop_callback], log_every_n_steps=10)
        trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='sl')
    parser.add_argument('--fithouses', type=str, default='ukdale15')
    args = parser.parse_args()
    # train
    train(args.method, args.fithouses)
    