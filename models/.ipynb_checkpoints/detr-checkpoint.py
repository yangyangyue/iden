import sys
sys.path.append('..')
from dataset import WINDOW_SIZE
import torch
from torch import nn
import torch.nn.functional as F

"""
detr
没有使用匈牙利算法，感觉不太需要
"""

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

class DetrNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=128, n_class=2, num_queries=16):
        super().__init__()
        self.num_queries = num_queries
        self.cnn = BottleNeck(in_channels, out_channels)
        self.transformer = nn.Transformer(d_model=out_channels, nhead=2, dim_feedforward=512, batch_first=True, num_encoder_layers = 1, num_decoder_layers = 1)
        self.query_embed = nn.Embedding(num_queries, out_channels)
        self.pos_embed_encoder = nn.Parameter(torch.randn(WINDOW_SIZE, out_channels))  # For Encoder
        self.pos_embed_decoder = nn.Parameter(torch.randn(num_queries, out_channels))  # For Decoder
        self.fc_clz = nn.Linear(out_channels, 1+n_class)
        self.fc_pos = nn.Linear(out_channels, 1)
        self.cce = nn.CrossEntropyLoss()
        
    def forward(self, ids, stamps, aggs, gt_poses, gt_clzes):
        B, L = aggs.shape  # Batch size, sequence length
        features = self.cnn(aggs[:, None, :]).permute(0, 2, 1)  # (B, L, out_channels)
        
        encoder_input = features + self.pos_embed_encoder  # (B, L, out_channels)
        
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        decoder_input = queries + self.pos_embed_decoder # (B, num_queries, out_channels)
        
        out = self.transformer(encoder_input, decoder_input)  # (B, num_queries, out_channels)
        
        # Predictions
        pred_poses = torch.sigmoid(self.fc_pos(out)).squeeze(-1)  # (B, num_queries)
        pred_logits = self.fc_clz(out)  # (B, num_queries, num_classes + 1)

        if self.training:
            loss_pos, loss_clz = 0, 0
            for w_pred_poses, w_pred_logits, w_gt_poses, w_gt_clzes in zip(pred_poses, pred_logits, gt_poses, gt_clzes):
                w_gt_poses = w_gt_poses/WINDOW_SIZE
                matched = []
                dists = torch.abs(w_gt_poses[:, None] - w_pred_poses[None, :])  # (Q, M)
                for i, dis in enumerate(dists):
                    dis = dis.clone()
                    dis[matched] = torch.inf
                    matched.append(dis.argmin().item())
                    if len(matched) == len(w_pred_poses):       
                        break
                unmatched = [idx for idx in range(len(w_pred_poses)) if idx not in matched]
                matched_poses, matched_logits, unmatched_logits = w_pred_poses[matched], w_pred_logits[matched], w_pred_logits[unmatched]
                if len(matched) > 0: loss_clz += self.cce(matched_logits, w_gt_clzes)
                if len(unmatched) > 0: loss_clz += self.cce(unmatched_logits, torch.zeros(len(unmatched_logits), dtype=torch.long).to(unmatched_logits.device))
                if len(matched) > 0: loss_pos += F.smooth_l1_loss(matched_poses, w_gt_poses)
            total_loss = loss_clz + 10000 * loss_pos
            return total_loss / len(gt_poses)
        else:
            pred_poses = (pred_poses * WINDOW_SIZE).clamp(0, WINDOW_SIZE - 1).long()  # 还原到真实坐标
            pred_classes = torch.argmax(pred_logits, dim=2)  
            mask = pred_classes > 0  # (B, num_queries)
            return [p[m] for p, m in zip(pred_poses, mask)], [c[m] for c, m in zip(pred_classes, mask)]
