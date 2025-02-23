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

def test(method, test_house, fit_houses):
    # init
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('high')
    sets = get_sets(test_house, 'test')
    for app_abb, app_set in zip('kmdwf', sets):
        save_path = Path('results') / f'{method}-{fit_houses}-{test_house}-{app_abb}.csv'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        datamodule = NilmDataModule(test_set=app_set, bs=256)
        ckpt_files = list(Path('checkpoints').expanduser().glob(f'{method}-{fit_houses}-{app_abb}*'))
        model = NilmNet.load_from_checkpoint(ckpt_files[-1], method=method, save_path=save_path)
        trainer = pl.Trainer(devices="auto", accelerator="auto")
        trainer.test(model, datamodule=datamodule, verbose=False)
        # break

if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='sl')
    parser.add_argument('--testhouse', type=str, default='ukdale2')
    parser.add_argument('--fithouses', type=str, default='ukdale15')
    args = parser.parse_args()
    # test
    test(args.method, args.testhouse, args.fithouses) 
    