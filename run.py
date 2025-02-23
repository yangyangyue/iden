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


def run(method, house):
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
        early_stop_callback = EarlyStopping(monitor="f1", mode="max", patience=10)
        trainer = pl.Trainer(devices="auto", accelerator="auto", max_epochs=30, callbacks=[checkpoint_callback, early_stop_callback], log_every_n_steps=10, num_sanity_val_steps=0)
        trainer.fit(model, datamodule=datamodule)
        save_path = Path('results') / f'{method}-{fit_houses}-{test_house}-{app_abb}.csv'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        datamodule = NilmDataModule(test_set=app_set, bs=256)
        ckpt_files = list(Path('checkpoints').expanduser().glob(f'{method}-{fit_houses}-{app_abb}*'))
        model = NilmNet.load_from_checkpoint(ckpt_files[-1], method=method, save_path=save_path)
        trainer = pl.Trainer(devices="auto", accelerator="auto")
        trainer.test(model, datamodule=datamodule, verbose=False)
        
if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='sl')
    parser.add_argument('--house', type=str, default='ukdale2')
    args = parser.parse_args()
    # train
    run(args.method, args.house)
    