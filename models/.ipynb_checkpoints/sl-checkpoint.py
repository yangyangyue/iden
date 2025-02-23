import sys
sys.path.append('..')
from dataset import WINDOW_SIZE
import torch
from torch import nn
import torch.nn.functional as F

"""
用序列标注的方式识别事件
无事件点标注为0 开事件点标注为1 关事件点标注为2
"""

"""
yolo的做法是多尺度锚框，也就是铺一大堆锚框，然后拟合
在我的实验中，事件长度是固定的，那么一个做法是固定锚框长度，不再需要拟合长度，另一个做法是以点的形式

yolo中输出的是每个位置每个预设锚框的类型和偏移
在我这个场景下，每个位置仅预设一个锚框，且偏移不再需要，那么我的输出就是每个位置的类别，这实际上和序列标注是一样的了，所以不再需要做yolo相关实验，因为在我的场景下，其已经退化成序列标注了
"""

def ann2seq(poses, clzes, device):
    """ 基于事件标注生成序列标签 """
    seq = torch.zeros((len(poses), WINDOW_SIZE), dtype=torch.long).to(device)
    for batch_idx, (w_poses, w_clzes) in enumerate(zip(poses, clzes)):
        for pos, clz in zip(w_poses, w_clzes):
            seq[batch_idx, int(pos)] = clz
    return seq

def seq2ann(seqs, scores, thresh=6):
    """ 基于序列标签生成事件标注 """
    poses, clzes = [], []
    for w_seq, w_scores in zip(seqs, scores):
        w_poses = torch.nonzero(w_seq)[:, 0]
        w_clzes, w_scores = w_seq[w_poses], w_scores[w_poses]
        sorted_ids = torch.argsort(w_scores, descending=True)
        w_poses, w_clzes, w_scores = w_poses[sorted_ids], w_clzes[sorted_ids], w_scores[sorted_ids]
        keep = []
        for i, pos in enumerate(w_poses):
            if all(abs(pos - w_poses[k]) > thresh for k in keep):
                keep.append(i)
        poses.append(w_poses[keep])
        clzes.append(w_clzes[keep])
    return poses, clzes

class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        mid_channels = out_channels // 4
        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
            nn.Conv1d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        if self.in_channels == self.out_channels: x = x + self.bottleneck(x)
        else: x = self.bottleneck(x)
        return x.relu()


class SlNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=256, n_class=2):
        super().__init__()
        self.cnn = nn.BottleNeck(
            BottleNeck(in_channels, out_channels),
            BottleNeck(out_channels, out_channels),
            BottleNeck(out_channels, out_channels),
            BottleNeck(out_channels, out_channels),
        )
        self.pos_embed = nn.Parameter(torch.randn(WINDOW_SIZE, out_channels))
        layer = nn.TransformerEncoderLayer(d_model=out_channels, nhead=2, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, 1)
        self.out = nn.Conv1d(out_channels, 1 + n_class, kernel_size=3, stride=1, padding=1)
        self.cce = nn.CrossEntropyLoss()

    def forward(self, ids, stamps, aggs, poses, clzes):
        features = self.cnn(aggs[:, None, :]).permute(0, 2, 1) # (bs, L， out_channels)
        features = self.transformer(features+self.pos_embed).permute(0, 2, 1) # (bs, out_channels, L)
        logits = self.out(features)  # (bs, 1+n_class, L)
        return self.cce(logits, ann2seq(poses, clzes, aggs.device)) if self.training else seq2ann(torch.argmax(logits, dim=1), F.softmax(logits, dim=1).max(dim=1).values)

