import lightning as L
import torch

from torch import FloatTensor, LongTensor, nn, Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from transformers import PretrainedConfig

from gen.transformer import MultiheadAttention, attention
from metrics import cal_metrics


class GmimeConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`NougatModel`]. It is used to
    instantiate a Nougat model according to the specified arguments, defining the model architecture

    Args:
        window_length:
            Length of sliding window
        hidden_dimension:
            Demension of feature map
    """
    model_type = "gmime"

    def __init__(
        self,
        window_length: int = 2048,
        n_class: int = 3,
        hidden_dimension: int = 512,
        lr: float = 1e-4,
        warmup_steps: float = 0,
        min_lr: float = 1e-5,
        gamma: float = 1,
        lr_step: float = 1,
        **kwargs,
    ):
        super().__init__()
        self.window_length = window_length
        self.n_class = n_class
        self.hidden_dimension = hidden_dimension
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.gamma = gamma
        self.lr_step = lr_step


class CNN_Module(nn.Module):
    """
    CNN module is used to extract local features from mains

    Args:
        hd: the dimension of extracted features (hidden features)
    """

    def __init__(self, hd):
        super(CNN_Module, self).__init__()
        self.hd = hd
        self.cnn = nn.Sequential(
            nn.Conv1d(1, hd // 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv1d(hd // 16, hd // 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv1d(hd // 8, hd // 4, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.01),
            nn.Conv1d(hd // 4, hd // 2, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.01),
            nn.Conv1d(hd // 2, hd, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(hd),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        return self.cnn(x)


class PositionEncoding(nn.Module):
    def __init__(self, in_channels, dropout=0.1, sine=True):
        super(PositionEncoding, self).__init__()
        if sine:
            self.position_embedding = self.PositionEmbeddingSine(in_channels, dropout)
        else:
            self.position_embedding = self.PositionEmbeddingLearned(in_channels)

    def forward(self, x):
        # x: shape (bs, out_channels, L); self.position_embedding(x):shape (out_channels, L)
        return self.position_embedding(x)

    # 基于sin函数的位置编码
    class PositionEmbeddingSine(nn.Module):
        def __init__(self, in_channels, dropout=0.1):
            # in_channels是（特征图）通道数，length是（特征图）长度，用于确定位置编码的大小
            super(PositionEncoding.PositionEmbeddingSine, self).__init__()
            self.in_channels = in_channels
            self.dropout = nn.Dropout(p=dropout)

        def forward(self, x):
            assert self.in_channels % 2 == 0, "位置编码要求通道为复数"
            length = x.shape[-2]
            pe = torch.zeros(length, self.in_channels)  # 存储位置编码
            position = torch.arange(0, length).unsqueeze(1).float()  # 存储词与词的绝对位置
            # 计算每个通道的位置编码的分母部分
            # n^{d/(2i)}  i<self.in_channels // 2
            div_term = (
                torch.full([1, self.in_channels // 2], 10000)
                .pow((torch.arange(0, self.in_channels, 2) / self.in_channels).float())
                .float()
            )
            # 偶数通道使用sin, 奇数通道使用cos
            pe[:, 0::2] = torch.sin(position / div_term)
            pe[:, 1::2] = torch.cos(position / div_term)
            return self.dropout(pe.to(x.device))

    # 基于学习的位置编码
    class PositionEmbeddingLearned(nn.Module):
        def __init__(self, n_dim, length=1024, dropout=0.1):
            super(PositionEncoding.PositionEmbeddingLearned, self).__init__()
            self.n_dim = n_dim
            self.length = length
            self.dropout = nn.Dropout(p=dropout)
            self.embed = nn.Embedding(length, n_dim)
            nn.init.uniform_(self.embed.weight)

        def forward(self, x):
            i = torch.arange(self.length).to(x.device)
            pe = self.embed(i)
            return self.dropout(pe)


class EventGenerator:
    """implement greedy search function for event generation"""

    def __init__(self):
        super(EventGenerator, self).__init__()

    def prepare(self, in_clzes: Tensor, in_poses: Tensor) -> Tensor:
        """
        Combine classes and poses of event sequence to build input to decoder.

        Args:
            in_clzes: (N, T), the sequence of cls used to build prompt.
            in_poses: (N, T), the sequence of pos used to build prompt.
        Returns:
            (N, T, 2), the prompt fed to the decoder.
        """
        return torch.stack([in_clzes, in_poses], dim=-1)

    def next(self, inputs, hidden_feature, mask=None, pad_mask=None):
        raise NotImplementedError("A specific model must to implement method `next()`")

    def greedy_search(
        self,
        in_clzes: LongTensor = None,
        in_poses: FloatTensor = None,
        end_cls: int = None,
        hf: Tensor = None,
        max_length: int = 24,
    ):
        r"""
        use greedy search method generate the whole output from given input

        Args:
            in_clzes: (N, T), the sequence of cls used to build prompt.
            in_poses: (N, T), the sequence of pos used to build prompt.
            end_cls: the class id of <eos>.
            hf: (N, S, hd), the hidden features extracted by encoder from mains.
            max_length: limit the length of outputs.
        """
        device = in_clzes.device
        bs = in_clzes.shape[0]
        # keep track of which sequences are already finished
        unfinished = torch.ones(bs, dtype=torch.long, device=device)
        # ensure positions are generated in order
        last_pos = torch.zeros(bs, dtype=torch.long, device=device)
        while True:
            # 1. combine in_clzes and in_poses to build prompt
            inputs = self.prepare(in_clzes, in_poses)

            # 2. config tgt_mask and pad_mask
            T = inputs.shape[1]
            mask = torch.triu(torch.ones(T, T, device=device)) == 1
            mask = mask.transpose(0, 1).float()
            mask = mask.masked_fill(mask == 0, float("-inf"))
            mask = mask.masked_fill(mask == 1, float(0.0))
            pad_mask = inputs[:, :, 0] == 0

            # 3. generate next_cls and next_pos
            cls, pos = self.next(inputs, hf, mask=mask, pad_mask=pad_mask)
            next_cls_logits, next_pos_logits = (cls[:, -1, :], pos[:, -1, :])
            next_cls = torch.argmax(next_cls_logits, dim=-1)
            next_pos = torch.argmax(next_pos_logits, dim=-1)
            exceed_mask = next_pos <= last_pos
            next_cls[exceed_mask] = end_cls
            last_pos = next_pos
            next_cls, next_pos = next_cls * unfinished, next_pos * unfinished

            # 4. update inputs
            in_clzes = torch.cat([in_clzes, next_cls[:, None]], dim=-1)
            in_poses = torch.cat([in_poses, next_pos[:, None]], dim=-1)

            # 5. whether finish
            unfinished = unfinished.mul(next_cls != end_cls)
            if unfinished.max() == 0 or in_clzes.shape[-1] > max_length:
                return in_clzes[:, 1:], in_poses[:, 1:]


class Gmime(L.LightningModule, EventGenerator):
    def __init__(self, config: GmimeConfig):
        super(Gmime, self).__init__()
        self.config = config
        self.cnn = CNN_Module(self.config.hidden_dimension)
        self.tokenizer = nn.Linear(2, self.config.hidden_dimension)
        self.positionEncoding = PositionEncoding(self.config.hidden_dimension)
        self.transformer = nn.Transformer(
            d_model=self.config.hidden_dimension,
            num_encoder_layers=2,
            num_decoder_layers=2,
            batch_first=True,
        )
        self.classify = nn.Linear(self.config.hidden_dimension, self.config.n_class + 1)
        self.localize = MultiheadAttention(self.config.hidden_dimension, 8)
        # loss
        self.bce = nn.BCELoss()
        self.cce = nn.CrossEntropyLoss()
        self.center_loss = nn.CrossEntropyLoss()
        self.width_loss = nn.SmoothL1Loss()
        self.l1 = nn.L1Loss()
        self.end_clz = 0
        self.validation_outputs = [[], [], [], [], []]

    def next(self, inputs, hf, mask=None, pad_mask=None) -> tuple[Tensor, Tensor]:
        # long long ago, a sb used conv1d here to implement rather than tokenizer
        y = self.tokenizer(inputs.float())
        y = y + self.positionEncoding(y)
        y = self.transformer.decoder(
            y, hf, tgt_mask=mask, tgt_key_padding_mask=pad_mask
        )
        return self.classify(y), attention(y, hf, hf)[1]

    def shift(self, tgt: Tensor):
        shifted = torch.zeros_like(tgt)
        shifted[..., 0] = -1
        shifted[..., 1:] = tgt[..., :-1].clone()
        return shifted

    def training_step(self, batch, batch_idx):
        """
        stamps_batch: list(np.ndarray)  N, L, timestamps of the each item in the batch.
        powers_batch: Tensor (N, 1, L), powers of the each item in the batch.
        poses_batch: list(Tensor) N T, event positions of each item in the batch.
        clzes_batch: list(Tensor) N T, event classes of each item in the batch.
        """
        stamps_batch, powers_batch, poses_batch, clzes_batch = batch
        # 1. get hidden_feature through encoder
        x = powers_batch[:, None, :].to(self.device)
        x = self.cnn(x).permute(0, 2, 1)
        x = x + self.positionEncoding(x)
        hf = self.transformer.encoder(x)

        # 2. prepare the input and output of decoder
        end_pos, end_clz = torch.tensor([0]), torch.tensor([0])
        poses_batch = [torch.cat([poses, end_pos]) for poses in poses_batch]
        clzes_batch = [torch.cat([clzes, end_clz]) for clzes in clzes_batch]
        tgt_poses = pad_sequence(clzes_batch, batch_first=True, padding_value=0)
        tgt_clses = pad_sequence(poses_batch, batch_first=True, padding_value=0)
        tgt_poses, tgt_clses = tgt_poses.to(self.device), tgt_clses.to(self.device)

        shifted_clses, shifted_poses = self.shift(tgt_clses), self.shift(tgt_poses)
        prompts = self.prepare(shifted_clses, shifted_poses)

        # 3. config tgt_mask and pad_mask
        T = prompts.shape[1]
        mask = torch.triu(torch.ones(T, T, device=self.device)) == 1
        mask = mask.transpose(0, 1).float()
        mask = mask.masked_fill(mask == 0, float("-inf"))
        mask = mask.masked_fill(mask == 1, float(0.0))
        pad_mask = shifted_clses == self.end_clz

        # 4. get pred events and then calculate the loss
        clzes, poses = self.next(prompts, hf, mask=mask, pad_mask=pad_mask)
        loss1 = self.cce(clzes.reshape(-1, clzes.shape[-1]), tgt_clses.reshape(-1))
        loss2 = self.cce(poses.reshape(-1, poses.shape[-1]), tgt_poses.reshape(-1))
        return loss1 + loss2

    def validation_step(self, batch, batch_idx):
        stamps_batch, powers_batch, poses_batch, clzes_batch = batch
        N = len(clzes_batch)
        """
        stamps_batch: list(np.ndarray)  N, L, timestamps of the each item in the batch.
        powers_batch: Tensor (N, 1, L), powers of the each item in the batch.
        poses_batch: list(Tensor) N T, event positions of each item in the batch.
        clzes_batch: list(Tensor) N T, event classes of each item in the batch.
        """
        # 1. get hidden_feature through encoder
        x = powers_batch[:, None, :].to(self.device)
        x = self.cnn(x).permute(0, 2, 1)
        x = x + self.positionEncoding(x)
        hf = self.transformer.encoder(x)

        in_clzes = torch.tensor([-1]).tile((N, 1)).to(self.device)
        in_poses = torch.tensor([0]).tile((N, 1)).to(self.device)
        clzes, poses = self.greedy_search(in_clzes, in_poses, self.end_clz, hf)

        # 2. record the output
        self.validation_outputs[0].extend(clzes_batch)
        self.validation_outputs[1].extend(poses_batch)
        for clzes_, poses_ in zip(clzes, poses):
            self.validation_outputs[2].append(clzes_[clzes_ != self.end_clz])
            self.validation_outputs[3].append(poses_[clzes_ != self.end_clz])
        self.validation_outputs[4].extend([powers_batch[i] for i in range(N)])

    def on_validation_epoch_end(self):
        tp, fp, fn, pre, rec, f1 = cal_metrics(*self.validation_outputs)
        self.log("tp", tp)
        self.log("fp", fp)
        self.log("fn", fn)
        self.log("pre", pre)
        self.log("rec", rec)
        self.log("f1", f1)
        self.validation_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr)
        scheduler = {
            "scheduler": self.exponential_scheduler(
                optimizer,
                self.config.warmup_steps,
                self.config.lr,
                self.config.min_lr,
                self.config.gamma,
            ),
            "name": "learning_rate",
            "interval": "step",
            "frequency": self.config.lr_step,
        }
        return [optimizer], [scheduler]

    @staticmethod
    def exponential_scheduler(optimizer, warmup_steps, lr, min_lr=5e-5, gamma=0.9999):
        def lr_lambda(x):
            if x > warmup_steps or warmup_steps <= 0:
                if lr * gamma ** (x - warmup_steps) > min_lr:
                    return gamma ** (x - warmup_steps)
                else:
                    return min_lr / lr
            else:
                return x / warmup_steps

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    def forward(self, x, targets, event_detect=False):
        ...
