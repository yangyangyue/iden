import json
import os
import tempfile
from typing import Union, List
import sys

sys.path.append("/home/aistudio/external-libraries")

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path


class BaseConfig:
    r"""
    This is the configuration class to store some basic configuration

    Args:
        epochs (`int`, *optional*, defaults to 1000):
            total number of training rounds.
        lr (`float`, *optional*, defaults to 1e-4):
            init learning rate
        lr_drop (`int`, *optional*, defaults to 400):
            interval for updating learning rate
        gama (`float`, *optional*, defaults to 1e-1):
            factor for updating learning rate
        batch_size (`int`, *optional*, defaults to 4):
            the size of each batch
        load_his (`bool`, *optional*, defaults to False):
            whether load checkpoint or train from scratch
    """

    def __init__(
        self,
        epochs=1000,
        lr=1e-4,
        lr_drop=400,
        gama=0.1,
        batch_size=4,
        load_his=False,
    ) -> None:
        self.epochs = epochs
        self.lr = lr
        self.lr_drop = lr_drop
        self.gama = gama
        self.batch_size = batch_size
        self.load_his = load_his

    def save(self, path: Path):
        with open(path, "w") as file:
            json.dump(self.__dict__, file, indent=4)

    @classmethod
    def load(cls, path: Path):
        with open(path, "r") as file:
            return cls(**json.load(file))


"""
由targets（包含样本中事件的起止位置和类别）生成按序列标注：
输出(L,N)
"""


def to_seq(targets, length, device, label_method="BIO"):
    ground = torch.zeros((len(targets), length), dtype=torch.long).to(device)
    for batch_idx, target in enumerate(targets):
        boxes = target["boxes"]
        labels = target["labels"]
        for box_idx, box in enumerate(boxes):
            if label_method == "BO":
                ground[batch_idx, int(box[0]) : int(box[1]) + 1] = labels[box_idx]
            elif label_method == "BIO":
                ground[batch_idx, int(box[0])] = 2 * labels[box_idx] - 1
                ground[batch_idx, int(box[0]) + 1 : int(box[1]) + 1] = (
                    2 * labels[box_idx]
                )
            elif label_method == "BILO":
                ground[batch_idx, int(box[0])] = 3 * labels[box_idx] - 2
                ground[batch_idx, int(box[0]) + 1 : int(box[1])] = (
                    3 * labels[box_idx] - 1
                )
                ground[batch_idx, int(box[1])] = 3 * labels[box_idx]
    return ground


"""
输入2个锚框序列，boxes1有N个框，boxes2有M个框
输出(N,M),iou[i,j]是boxes1中i框和boxes2中j框的iou
"""


def box_iou(boxes1, boxes2):
    area1 = boxes1[:, 1] - boxes1[:, 0]
    area2 = boxes2[:, 1] - boxes2[:, 0]
    lt = torch.max(boxes1[:, None, 0], boxes2[:, 0])  # left-top [N,M,2]
    rb = torch.min(boxes1[:, None, 1], boxes2[:, 1])  # right-bottom [N,M,2]
    inter = (rb - lt).clamp(min=0)  # [N,M]，inter[i,j]是boxes1中i框和boxes2中j框的相交区域，不相交设为0
    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def recording(power, pred, target, case_dir):
    power = power.cpu().detach().numpy()
    target = {k: v.cpu().detach().numpy().tolist() for k, v in target.items()}
    # pred = {k: v.cpu().detach().numpy().tolist() for k, v in pred.items()}
    plt.figure(figsize=(16, 10), dpi=300)
    plt.plot(power)
    max_power = max(power)
    for label, boxes in zip(pred["labels"], pred["boxes"]):
        plt.axvline(boxes[0].item(), ymin=max_power / 2, color="lightblue")
        plt.axvline(boxes[1].item(), ymin=max_power / 2, color="blue")
        plt.text(boxes[1].item(), max_power, str(label.item()), color="blue")
    for label, boxes in zip(target["labels"], target["boxes"]):
        plt.axvline(boxes[0], ymax=max_power / 2, color="lightcoral")
        plt.axvline(boxes[1], ymax=max_power / 2, color="red")
        plt.text(boxes[1], max_power / 2, str(label), color="red")
    plt.savefig(f"{case_dir}/{target['idx']}.png", bbox_inches="tight")
    plt.cla()


