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





"""
输入2个锚框序列，boxes1有N个框，boxes2有M个框
输出(N,M),iou[i,j]是boxes1中i框和boxes2中j框的iou
"""


def iou(boxes1, boxes2):
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


