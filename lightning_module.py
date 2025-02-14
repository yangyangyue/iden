"""
MIT License
Copyright (c) 2024-present lily.

written by lily
email: lily231147@gmail.com
"""

import sys
from dataset import vars

import lightning as L
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from models.mada import MadaNet
from models.mvae import MvaeNet
from models.vae import VaeNet
from models.s2p import S2pNet
from models.s2s import S2sNet

class NilmNet(L.LightningModule):
    """
    所有识别模型输出两个序列：
    pred_poses: (N, S) 识别的位置
    pred_clzes: (N, S) 识别的类型
    """
    def __init__(self, method, save_path = None) -> None:
        super().__init__()
        self.method = method
        self.save_path = save_path
        if method == 'mada':
            self.model = MadaNet()
        elif method == 'vae':
            self.model = VaeNet()
        elif method == 's2p':
            self.model = S2pNet()
        elif method == 's2s':
            self.model = S2sNet()
        elif method == 'mvae':
            self.model = MvaeNet()
        self.tp, self.fp, self.fn = 0, 0, 0
    
    def forward(self, ids, stamps, aggs, poses=None, clzes=None):
        return self.model(ids, stamps, aggs, poses, clzes)
        
    def training_step(self, batch, _):
        #  aggs: (N, L) 其他变量都是元组 元组中每一项对应于一个窗口的数据
        ids, stamps, aggs, poses, clzes = batch
        poses = [p.to(aggs.device) for p in poses]
        clzes = [c.to(aggs.device) for c in clzes]
        loss = self(ids, stamps, aggs, poses, clzes)
        self.losses.append(loss.item())
        return loss
        
    def on_train_epoch_end(self) -> None:
        self.log('loss', np.mean(self.losses), on_epoch=True, prog_bar=True, logger=True)
        self.losses.clear()
    
    def validation_step(self, batch, _):
        ids, stamps, aggs, poses, clzes = batch
        poses = [p.to(aggs.device) for p in poses]
        clzes = [c.to(aggs.device) for c in clzes]
        pred_poses, pred_clzes = self(ids, stamps, aggs)
        b_tp, b_fp, b_fn = self.cal_metrics(poses, clzes, pred_poses, pred_clzes)
        self.tp += b_tp
        self.fp += b_fp
        self.fn += b_fn
    
    def on_validation_epoch_end(self):
        pre = self.tp / (self.tp + self.fp) if self.tp + self.fp > 0 else 0
        rec = self.tp / (self.tp + self.fn) if self.tp + self.fn > 0 else 0
        f1 = 2.0 * pre * rec / (pre + rec) if pre + rec > 0 else 0
        self.log('pre', pre, on_epoch=True, prog_bar=True, logger=True)
        self.log('rec', rec, on_epoch=True, prog_bar=True, logger=True)
        self.log('f1', f1, on_epoch=True, prog_bar=True, logger=True)
        self.tp, self.fp, self.fn = 0, 0, 0
    
    def test_step(self, batch, _):
        ids, stamps, aggs, poses, clzes = batch
        poses = [p.to(aggs.device) for p in poses]
        clzes = [c.to(aggs.device) for c in clzes]
        pred_poses, pred_clzes = self(ids, stamps, aggs)
        b_tp, b_fp, b_fn = self.cal_metrics(poses, clzes, pred_poses, pred_clzes)
        self.tp += b_tp
        self.fp += b_fp
        self.fn += b_fn

    def on_test_epoch_end(self):
        pre = self.tp / (self.tp + self.fp) if self.tp + self.fp > 0 else 0
        rec = self.tp / (self.tp + self.fn) if self.tp + self.fn > 0 else 0
        f1 = 2.0 * pre * rec / (pre + rec) if pre + rec > 0 else 0
        self.log('pre', pre, on_epoch=True, prog_bar=True, logger=True)
        self.log('rec', rec, on_epoch=True, prog_bar=True, logger=True)
        self.log('f1', f1, on_epoch=True, prog_bar=True, logger=True)
        self.tp, self.fp, self.fn = 0, 0, 0
        self.print2file('pre', pre, 'rec', rec, 'f1', f1)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters())
        scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
        return [optimizer], [scheduler]

    def print2file(self, *args):
        with open('test_results.log', 'a') as f:
            sys.stdout = f
            print(self.save_path.stem, *args)
            sys.stdout = sys.__stdout__

    def cal_metrics(gt_poses, gt_clzes, pred_poses, pred_clzes, thresh: int = 3):
        """ 计算 tp, fp, fn, pre, rec, f1 """
        tp, fp, fn = 0, 0, 0 
        # w_的含义是当前滑动窗口 
        for (w_gt_poses, w_gt_clzes, w_pred_poses, w_pred_clzes) in zip(gt_poses, gt_clzes, pred_poses, pred_clzes):
            w_tp, w_fp, w_fn = 0, 0, 0
            for clz in (1, 2):
                # c_的含义是当前事件类型
                wc_gt_poses = w_gt_poses[w_gt_clzes == clz]
                wc_pred_poses = w_pred_poses[w_pred_clzes == clz]
                if len(wc_gt_poses) == 0: w_fp += len(wc_pred_poses)
                elif len(wc_pred_poses) == 0: w_fn += len(wc_gt_poses)
                else:
                    distanes = (wc_gt_poses[:, None] - wc_pred_poses[None, :]).abs()
                    matched = []
                    for dis in distanes:
                        dis[matched] = thresh + 1
                        if dis.min().item() <= thresh:
                            matched.append(dis.argmin().item())
                    w_tp += len(matched)
                    w_fp += len(wc_pred_poses) - len(matched)
                    w_fn += len(wc_gt_poses) - len(matched)
            tp += w_tp
            fp += w_fp
            fn += w_fn
        return tp, fp, fn

def collate_fn(batch):
    """因为每个窗口的事件数量不一致，所以需要手动整理"""
    ids, stamps, aggs, poses, clzes = tuple(zip(*batch))
    return ids, stamps, torch.stack([torch.as_tensor(agg, dtype=torch.float32) for agg in aggs]), [torch.as_tensor(w_poses, dtype=torch.float32) for w_poses in poses], [torch.as_tensor(w_clzes, dtype=torch.float32) for w_clzes in clzes]

class NilmDataModule(L.LightningDataModule):
    def __init__(self, train_set=None, val_set=None, test_set=None, bs=256):
        super().__init__()
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.batch_size = bs

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=18, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=18, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=18, collate_fn=collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=18, collate_fn=collate_fn)