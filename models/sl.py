import sys
sys.path.append('..')
from dataset import WINDOW_SIZE
import torch
from torch import nn

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

def seq2ann(seqs):
    """ 基于序列标签生成事件标注 """
    poses, clzes = [], []
    for batch_idx, w_seq in enumerate(seqs):
        w_poses = torch.nonzero(w_seq)[:, 0]
        w_clzes = w_seq[w_poses]
        poses.append(w_poses)
        clzes.append(w_clzes)
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


class PositionEmbeddingSine(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        assert channels % 2 == 0, "位置编码通道数必须是偶数"

        self.channels = channels
        self.dropout = nn.Dropout(p=dropout)

        # 生成位置编码
        pe = torch.zeros(WINDOW_SIZE, self.channels)  # (L, D)
        position = torch.arange(0, WINDOW_SIZE, dtype=torch.float32).unsqueeze(1)  # (L, 1)
        div_term = torch.pow(10000, torch.arange(0, self.channels, 2).float() / self.channels)  # (D//2,)
        pe[:, 0::2] = torch.sin(position / div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position / div_term)  # 奇数维度
        self.register_buffer("pe", pe) 

    def forward(self, x):
        return self.dropout(x + self.pe.to(x.device))  # 位置编码 + 输入特征



class SlNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=512, n_class=2):
        super().__init__()
        self.cnn = BottleNeck(in_channels, out_channels)
        self.embedding = PositionEmbeddingSine(out_channels, WINDOW_SIZE)
        layer = nn.TransformerEncoderLayer(d_model=out_channels, nhead=8, dim_feedforward=800, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, 2)
        self.out = nn.Conv1d(out_channels, 1 + n_class, kernel_size=3, stride=1, padding=1)
        self.cce = nn.CrossEntropyLoss(weight=torch.Tensor([0.0001] + [1 for _ in range(n_class)]))

    def forward(self, ids, stamps, aggs, poses, clzes):
        features = self.cnn(aggs[:, None, :]) # (bs, out_channels, L)
        features = self.transformer(self.embedding(features).permute(0, 2, 1)).permute(0, 2, 1) # (bs, out_channels, L)
        logits = self.out(features)  # (bs, 1+n_class, L)
        return self.cce(logits, ann2seq(poses, clzes, aggs.device, WINDOW_SIZE)) if self.training else seq2ann(torch.argmax(logits, dim=1))

