import torch
from torch import nn
from utils import BaseConfig, to_seq

class SlConfig(BaseConfig):
    r"""
    This is the configuration class to store specific configuration of SlNet
    
    Args:
        in_channels (`int`, *optional*, defaults to 1):
            the channel of input feed to the network
        out_channels (`int`, *optional*, defaults to 400):
            the channel of feature map
        length (`int`, *optional*, defaults to 1024):
            sliding window length
        backbone (`str`, *optional*, defaults to `attention`):
            specific the temporal feature extraction module, choosing from `attention`, `lstm`, and `empty`
        label_method (`str`, *optional*, defaults to `BIO`):
            the method of sequence labeling
    """
    def __init__(
        self, 
        epochs=1200, 
        lr=0.0001, 
        lr_drop=800, 
        gama=0.1, 
        batch_size=4, 
        load_his=False,
        in_channels=1,  
        out_channels=400, 
        length=1024, 
        backbone="attention",
        label_method="BIO",
        ) -> None:
        super().__init__(epochs, lr, lr_drop, gama, batch_size, load_his)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.length = length
        self.backbone = backbone
        self.label_method=label_method


class CNN_Module(nn.Module):
    """CNN提取局部特征"""

    def __init__(self, in_channels, out_channels):
        super(CNN_Module, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cnn = nn.Sequential(
            nn.Conv1d(self.in_channels, self.out_channels // 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.out_channels // 8, self.out_channels // 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.out_channels // 4, self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(self.out_channels))

    def forward(self, x):
        return self.cnn(x)


"""
位置编码
1. 基于sin的位置编码
2. 基于学习的位置编码
"""


class PositionEncoding(nn.Module):
    def __init__(self, in_channels, length, dropout=0.1, sine=True):
        super(PositionEncoding, self).__init__()
        if sine:
            self.position_embedding = PositionEmbeddingSine(in_channels, length, dropout)
        else:
            self.position_embedding = PositionEmbeddingLearned(in_channels, length, dropout)

    def forward(self, x):
        # x: shape (bs, out_channels, L); self.position_embedding(x):shape (out_channels, L)
        return self.position_embedding(x)


# 基于sin函数的位置编码
class PositionEmbeddingSine(nn.Module):
    def __init__(self, in_channels, length, dropout=0.1):
        # in_channels是（特征图）通道数，length是（特征图）长度，用于确定位置编码的大小
        super(PositionEmbeddingSine, self).__init__()
        self.in_channels = in_channels
        self.length = length
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        assert self.in_channels % 2 == 0, "位置编码要求通道为复数"
        pe = torch.zeros(self.length, self.in_channels)  # 存储位置编码
        position = torch.arange(0, self.length).unsqueeze(1).float()  # 存储词与词的绝对位置
        # 计算每个通道的位置编码的分母部分
        # n^{d/(2i)}  i<self.in_channels // 2
        div_term = torch.full([1, self.in_channels // 2], 10000).pow(
            (torch.arange(0, self.in_channels, 2) / self.in_channels).float()).float()
        # 偶数通道使用sin, 奇数通道使用cos
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)
        # pe = pe.unsqueeze(0)
        return self.dropout(pe.to(x.device)).permute(1, 0)


# 基于学习的位置编码
class PositionEmbeddingLearned(nn.Module):
    def __init__(self, n_dim, length, dropout=0.1):
        super().__init__()
        self.n_dim = n_dim
        self.length = length
        self.dropout = nn.Dropout(p=dropout)
        self.embed = nn.Embedding(length, n_dim)
        nn.init.uniform_(self.embed.weight)

    def forward(self, x):
        i = torch.arange(self.length).to(x.device)
        pe = self.embed(i).unsqueeze(0)
        return self.dropout(pe)


"""
TransformerEncoder模块
"""


class Transformer(nn.Module):
    def __init__(self, d_model, nhead):
        super(Transformer, self).__init__()
        self.layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(self.layer, 4)

    def forward(self, x):
        # Transformer编码，输入之前要转置一下，输出时也要转置一下
        return self.encoder(x.permute(0, 2, 1)).permute(0, 2, 1)


########################################################################################

"""
下游任务网络: 对事件进行分类并输出锚框
"""


# 对特征图做后处理，转换成需要的形式
class OutBIO(nn.Module):
    def __init__(self, in_channels, num_classes, label_method):
        super(OutBIO, self).__init__()
        self.num_classes = num_classes
        if label_method == "BI":
            # O+(B-事件类别数)
            self.identify = nn.Conv1d(in_channels, 1 + self.num_classes, kernel_size=3, stride=1, padding=1)
        elif label_method == "BIO":
            # O+(B-事件类别数)
            self.identify = nn.Conv1d(in_channels, 1 + 2 * self.num_classes, kernel_size=3, stride=1, padding=1)
        elif label_method == "BILO":
            self.identify = nn.Conv1d(in_channels, 1 + 3 * self.num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.identify(x)


class PostProcessor(nn.Module):
    def __init__(self, label_method):
        super(PostProcessor, self).__init__()
        self.label_method = label_method

    # 将输出的标注序列转化为事件类别及位置
    def forward(self, seqs):
        device = seqs[0].device
        results = [{"boxes": [], "labels": []} for _ in range(len(seqs))]
        # calc
        if self.label_method == "BI":
            for batch_idx, seq in enumerate(seqs):
                edge_idxs = torch.nonzero(torch.diff(seq, prepend=torch.tensor([0]).to(device),
                                                     append=torch.tensor([0]).to(device))).squeeze()
                if len(edge_idxs) == 0:
                    continue
                labels = torch.index_select(seq, dim=0, index=edge_idxs[:-1])
                boxes = torch.stack((edge_idxs[:-1], edge_idxs[1:] - 1), dim=1)
                mask = labels > 0
                results[batch_idx]["labels"] = labels[mask].cpu().numpy()
                results[batch_idx]["boxes"] = boxes[mask].cpu().numpy()
        elif self.label_method == "BIO":
            diff = torch.diff(seqs,append=torch.zeros((len(seqs),1)).to(device))
            starts = torch.nonzero((seqs[:, :-3] % 2 == 1) & (diff[:, :-3] == 1) &
                                   (diff[:, 1:-2] == 0) & (diff[:, 2:-1] == 0))
            for row, col in starts:
                offset = torch.where(diff[row, col + 1:])[0][0] + 1
                label = torch.div(seqs[row, col] + 1, 2, rounding_mode='floor')
                results[row]["boxes"].append((col, col + offset))
                results[row]["labels"].append(label)
        elif self.label_method == "BILO":
            diff = torch.diff(seqs,append=torch.zeros((len(seqs),1)).to(device))
            starts = torch.nonzero((seqs[:, :-3] % 3 == 1) & (diff[:, :-3] == 1) & (diff[:, 1:-2] == 0) )
            for row, col in starts:
                offset = torch.where(diff[row, col + 1:])[0][0] + 2
                label = torch.div(seqs[row, col] + 1, 2, rounding_mode='floor')
                if seqs[row, col]+2 == seqs[row, col + offset]:
                    results[row]["boxes"].append((col, col + offset))
                    results[row]["labels"].append(label)
        return results


class SlNet(nn.Module):
    def __init__(self, in_channels, out_channels, length, num_classes, label_method, backbone):
        super(SlNet, self).__init__()
        self.with_attention = backbone == "with_attention"
        self.length = length
        self.label_method = label_method
        self.cnn = CNN_Module(in_channels, out_channels)
        self.positionEncoding = PositionEncoding(out_channels, length)
        self.transformer = Transformer(d_model=out_channels, nhead=8)
        self.outBIO = OutBIO(out_channels, num_classes, label_method)
        self.postProcessor = PostProcessor(label_method)
        # kettle: 0.002
        if label_method == "BI":
            self.cce = nn.CrossEntropyLoss(weight=torch.Tensor([0.0002] + [1 for _ in range(num_classes)]))
        elif label_method == "BIO":
            self.cce = nn.CrossEntropyLoss(weight=torch.Tensor([0.0002] + [1 for _ in range(2 * num_classes)]))
        elif label_method == "BILO":
            self.cce = nn.CrossEntropyLoss(weight=torch.Tensor([0.0002] + [1 for _ in range(3 * num_classes)]))
        self.is_training = False

    def forward(self, x, y):
        x = self.cnn(x)  # (bs, out_channels, L)
        if self.with_attention:
            x = x + self.positionEncoding(x)  # (bs, out_channels, L)
            x = self.transformer(x)  # (bs, out_channels, L)
        x = self.outBIO(x)  # (bs, 1+2*n_class, L) or (bs, 1+n_class, L)
        y = to_seq(y, self.length, x.device, self.label_method)  # (bs, L)
        loss, pred = self.cce(x, y), torch.argmax(x, dim=1)  # pred: (bs, L)
        return loss if self.training else self.postProcessor(pred)
