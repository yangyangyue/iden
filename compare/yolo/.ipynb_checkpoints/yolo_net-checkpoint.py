from typing import Tuple, List, Dict, Optional
from utils import BaseConfig

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from yolo.det import box_iou, Matcher, BalancedPositiveNegativeSampler, BoxCoder, clip_boxes_to_image, smooth_l1_loss, \
    remove_small_boxes, batched_nms

class YoloConfig(BaseConfig):
    r"""
    This is the configuration class to store specific configuration of YoloNet
    
    Args:
        in_channels (`int`, *optional*, defaults to 1):
            the channel of input feed to the network
        out_channels (`int`, *optional*, defaults to 400):
            the channel of feature map
        length (`int`, *optional*, defaults to 1024):
            sliding window length
        backbone (`str`, *optional*, defaults to 1024):
            specific the temporal feature extraction module, choosing from `attention`, `lstm`, and `empty`
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
        backbone="attention") -> None:
        super().__init__(epochs, lr, lr_drop, gama, batch_size, load_his)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.length = length
        self.backbone = backbone

class YoloNet(nn.Module):
    def __init__(self, in_channels, out_channels, length, num_classes=None,  backbone = None,
                 # 移除低目标概率
                 box_score_thresh=0.8, box_nms_thresh=0.3, box_detections_per_img=64,  # 测试时最终输出的框的个数
                 box_fg_iou_thresh=0.6, box_bg_iou_thresh=0.5,  # fast rcnn计算误差时，采集正负样本设置的阈值
                 # fast rcnn计算误差时采样的样本数，以及正样本占所有样本的比例，训练时使用
                 box_batch_size_per_image=128, box_positive_fraction=0.3,
                 bbox_reg_weights=None, class_shebei=10):
        super(YoloNet, self).__init__()
        self.with_attention = backbone == "with_attention"
        size = (2, 4, 8, 10, 12, 16, 32)
        aspect_ratio = (1.0,)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 8, kernel_size=(3,), padding=1),
            nn.ReLU(),
            nn.Conv1d(out_channels // 8, out_channels // 2, kernel_size=(3,), padding=1),
            nn.ReLU(),
            nn.Conv1d(out_channels // 2, out_channels, kernel_size=(3,), padding=1),
            nn.ReLU())
        self.pos_embedding = PositionEmbeddingSine(in_channels=out_channels, length=1024)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=out_channels, nhead=8, dim_feedforward=800), 2)
        self.anchor_generator = AnchorsGenerator(sizes=size, aspect_ratios=aspect_ratio)
        num_anchors = len(size) * len(aspect_ratio)
        self.roi_heads = RoIHeads(box_fg_iou_thresh, box_bg_iou_thresh, box_batch_size_per_image, box_positive_fraction,
                                  bbox_reg_weights, box_score_thresh, box_nms_thresh, box_detections_per_img, d_model,
                                  num_anchors, num_classes)

    def forward(self, images, targets=None):
        length = images.shape[-1]
        # 局部特征提取：CNN
        features = self.cnn(images)
        # # 全局特征提取：Transformer，需要先转置一下特征图，然后再转回来
        if self.with_attention:
            features = features + self.pos_embedding(features)
            features = features.permute(0, 2, 1)
            features = self.transformer(features)  # batch,query_num,d_model
            features = features.permute(0, 2, 1)  # batch,d_model,query_num
        # anchors是一个列表，每个元素对应于一个样本，是形状为(m*L),2的Tensor，表征m*L个锚框，m是为每个点预设的锚框数量
        anchors = self.anchor_generator(images, features)
        # 将rpn生成的数据以及标注target信息传入fast rcnn后半部分

        return self.roi_heads(features, anchors, length, targets)


################################################################################################

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


###################################################################################
"""
used to generate the start and the end index of each base anchor
"""


class AnchorsGenerator(nn.Module):

    def __init__(self, sizes=(128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0)):
        super(AnchorsGenerator, self).__init__()
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios

    def grid_anchors(self, grid_sizes, strides, cell_anchors):
        device = cell_anchors.device
        # shape: [grid_width] 对应原图上的x坐标(列)
        shift_x = torch.arange(0, grid_sizes, dtype=torch.float32, device=device) * strides
        # 计算anchors坐标(xmin, ymin, xmax, ymax)在原图上的坐标偏移量
        shifts = torch.stack([shift_x, shift_x], dim=1)
        # 将anchors模板与原图上的坐标偏移量相加得到原图上所有anchors的坐标信息(shape不同时会使用广播机制)
        shifts_anchor = shifts.view(-1, 1, 2) + cell_anchors.view(1, -1, 2)
        return shifts_anchor.reshape(-1, 2)

    """
    images: (bs,1,L)
    feature_maps: (bs,d_model,L)
    """

    def forward(self, images, feature_map):
        bs = feature_map.shape[0]  # bs = 4
        grid_size = feature_map.shape[-1]  # grid_sizes=1024
        image_size = images.shape[-1]  # image_size=1024
        dtype, device = feature_map.dtype, feature_map.device
        scales = torch.as_tensor(self.sizes, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(self.aspect_ratios, dtype=dtype, device=device)
        stride = torch.tensor(image_size / grid_size, dtype=torch.int64, device=device)  # strides=1
        # cell_anchors = tensor([[ -1.,1.],[ -2.,2.],[ -4.,4.],[ -5.,5.],[ -6.,6.], [ -8.,8.],[-16.,16.]]
        ws = (aspect_ratios[:, None] * scales[None, :]).view(-1)
        cell_anchors = (torch.stack([-ws, ws], dim=1) / 2).round()
        anchors_over_all_feature_maps = self.grid_anchors(grid_size, stride, cell_anchors)
        anchors = [anchors_over_all_feature_maps for _ in range(bs)]
        return anchors


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    # classification loss
    classification_loss = F.cross_entropy(class_logits, labels)
    sampled_pos_inds_subset = torch.where(torch.gt(labels, 0))[0]
    # 返回标签类别大于0位置的类别信息
    labels_pos = labels[sampled_pos_inds_subset]
    # shape=[num_proposal, num_classes]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 2)
    # regression loss
    box_loss = smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        size_average=False,
    ) / labels.numel()

    return classification_loss, box_loss


def permute_and_flatten(layer, N, A, C, W):
    """
    调整tensor顺序，并进行reshape
    Args:
        layer: 预测特征层上预测的目标概率或bboxes regression参数
        N: batch_size
        A: anchors_num_per_position
        C: classes_num
        W: length
        layer: 调整tensor顺序，并reshape后的结果[N, -1, C]
    """
    # view和reshape功能是一样的，先展平所有元素在按照给定shape排列
    # view函数只能用于内存中连续存储的tensor，permute等操作会使tensor在内存中变得不再连续，此时就不能再调用view函数
    # reshape则不需要依赖目标tensor是否在内存中是连续的
    # [batch_size, anchors_num_per_position * (C or 2), length]
    layer = layer.view(N, -1, C, W)
    # 调换tensor维度
    layer = layer.permute(0, 3, 1, 2)  # [N, W, -1, C]
    layer = layer.reshape(N, -1, C)
    return layer


def concat_box_prediction_layers(box_cls, box_regression, num_class):
    """
    translate the shape of box_cls and box_regression
    class_logits: (bs,num_anchors*num_classes,L)            ->  [
    box_regression: (bs, num_anchors * 2 * num_classes,L)   ->
    """
    box_cls_per_level = box_cls
    box_regression_per_level = box_regression
    # 遍历每个预测特征层
    # [batch_size, anchors_num_per_position * classes_num, length]
    # 注意，当计算RPN中的proposal时，classes_num=1,只区分目标和背景
    N, AxC, W = box_cls_per_level.shape
    # [batch_size, anchors_num_per_position * 2, length]
    Ax2 = box_regression_per_level.shape[1]
    # anchors_num_per_position
    A = Ax2 // (1 + num_class) // 2
    # classes_num
    C = 1 + num_class
    # [N, -1, C]
    box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, W)
    box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 2 * C, W)
    box_cls = []
    box_regression = []
    for i in range(N):
        box_cls.append(box_cls_per_level[i])
        box_regression.append(box_regression_per_level[i])
    return box_cls, box_regression


class RoIHeads(torch.nn.Module):

    def __init__(self, fg_iou_thresh, bg_iou_thresh, batch_size_per_image, positive_fraction,
                 bbox_reg_weights, score_thresh, nms_thresh, detection_per_img, in_channels, num_anchors, num_class):
        super(RoIHeads, self).__init__()
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False)
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)
        if bbox_reg_weights is None:
            bbox_reg_weights = (1, 1)
        self.box_coder = BoxCoder(bbox_reg_weights)
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detection_per_img = detection_per_img
        self.cls_pred = nn.Conv1d(in_channels, num_anchors * (1 + num_class), kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv1d(in_channels, num_anchors * 2 * (1 + num_class), kernel_size=1, stride=1)
        self.num_class = num_class

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        """
        assign gt_box for each proposal, except for:
            negative proposal: maximum iou < low_threshold)
            ignore proposal: low_threshold <= iou < high_threshold
        Return:
            matched_idxs: store index of matched gt_box for each proposal, setting it to 0 if no matched gt_box
            labels: store class of matched gt_box for each proposal, setting it 0 for negative proposal and -1 for
                    ignore proposal respectively
        """
        matched_idxs = []
        labels = []
        # handle proposals, gt_boxes and gt_labels of each image
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
            if gt_boxes_in_image.numel() == 0:
                # for negative image without gt_boxes, the class of all proposal is set to 0
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64,
                                                            device=device)
                labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
            else:
                # calculate iou between each gt_box and each proposal
                match_quality_matrix = box_iou(gt_boxes_in_image, proposals_in_image)
                # match corresponding gt_box index for each proposal by selecting the one with maximum iou,
                # set it to -1 where iou < low_threshold (negative proposal)
                # set it to -2 where low_threshold <= iou < high_threshold (ignore proposal)
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)
                # convert matched_idx to class of corresponding gt_box, whereas -1 and -2 is converted to class 0
                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)
                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image].to(dtype=torch.int64)
                # set the class of the negative proposal and ignore proposal to 0 and -1 respectively.
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1
            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels):
        """
        sample proposal to balance positive and negative ones, it will ignore proposals whose class is -1
        return:
            sampled_inds of shape `(bs, num_samples)`, the mask of selected positive and negative proposals
        """
        # for each image, select proposals with specific number(128), 30% of which are positive samples
        # sampled_pos_inds: (bs, num_positive_samples), sampled_neg_inds: (bs, num_negative_samples)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def select_training_samples(self, proposals, targets, class_logits, box_regression):
        dtype = proposals[0].dtype
        device = proposals[0].device
        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]
        # 为每个proposal匹配对应的gt_box，并划分到正负样本中
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # 按给定数量和比例采样正负样本
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)

        # 遍历每张图像
        for img_id in range(num_images):
            # 获取每张图像的正负样本索引
            img_sampled_inds = sampled_inds[img_id]
            # 获取对应正负样本的proposals信息
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            class_logits[img_id] = class_logits[img_id][img_sampled_inds]
            box_regression[img_id] = box_regression[img_id][img_sampled_inds]
            # 获取对应正负样本的真实类别信息
            labels[img_id] = labels[img_id][img_sampled_inds]
            # 获取对应正负样本的预测类别信息
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]
            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 2), dtype=dtype, device=device)
            # 获取对应正负样本的gt box信息
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        # 根据gt和proposal计算边框回归参数（针对gt的）
        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets, class_logits, box_regression

    def postprocess_detections(self, class_logits, box_regression, proposals, image_shape):
        """
        对网络的预测数据进行后处理，包括
            （1）根据proposal以及预测的回归参数计算出最终bbox坐标
            （2）对预测类别结果进行softmax处理
            （3）裁剪预测的boxes信息，将越界的坐标调整到图片边界上
            （4）移除所有背景信息
            （5）移除低概率目标
            （6）移除小尺寸目标
            （7）执行nms处理，并按scores进行排序
            （8）根据scores排序返回前topk个目标
        class_logits: (bs*m*L, (1+n_class)), the predict class vectors of all base anchors in one batch
        box_regression: (bs*m*L, 2 * (1+n_class)), similar to the class_logits
        proposals: (bs, m*L, 2), the indexes of each base anchors
        image_shapes=(L,L,L,L), the length of each image
        """
        device = class_logits.device
        # 预测目标类别数 :(1+n_class)
        num_classes = class_logits.shape[-1]
        # 获取每张图像的预测bbox数量 =m*L
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        # 根据proposal以及预测的回归参数计算出最终bbox坐标: (bs*m*L, 2)
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        # 对预测类别结果进行softmax处理
        pred_scores = F.softmax(class_logits, -1)
        # split boxes and scores per image
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        # 遍历每张图像预测信息
        for boxes, scores in zip(pred_boxes_list, pred_scores_list):
            # 裁剪预测的boxes信息，将越界的坐标调整到图片边界上
            boxes = clip_boxes_to_image(boxes, image_shape)
            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)
            # remove prediction with the background label
            # 移除索引为0的所有信息（0代表背景）
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 2)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            # remove low scoring boxes
            # 移除低概率目标，self.scores_thresh=0.05
            inds = torch.where(torch.gt(scores, self.score_thresh))[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
            # remove small boxes
            keep = remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            # non-maximun suppression, independently done per class
            keep = batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detection_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            all_boxes.append(torch.round(boxes).long())
            all_scores.append(scores)
            all_labels.append(labels)
        return all_boxes, all_scores, all_labels

    def cat_res(self, class_logits, box_regression):
        len_ = len(class_logits)
        if len_ == 1:
            return class_logits[0], box_regression[0]
        else:
            class_logits_ = class_logits[0]
            box_regression_ = box_regression[0]
            for k in range(1, len_):
                class_logits_ = torch.cat((class_logits_, class_logits[k]), dim=0)
                box_regression_ = torch.cat((box_regression_, box_regression[k]), dim=0)
            return class_logits_, box_regression_

    def forward(self, features, proposals, image_shape, targets=None):
        """
        Parameters:
            features of shape (bs, d_model, L)
            proposals of shape (7*L, 2)
            image_shapes of shape (L, L, L, L)
        """
        # class_logits:(bs,num_anchors*(1+num_classes),L)
        # box_regression:(bs, num_anchors * 2* (1+num_classes),L)
        class_logits = self.cls_pred(features)
        box_regression = self.bbox_pred(features)
        # reshape the class_logits and box_regression
        # class_logits:(bs, num_anchors * L ,(1+num_classes))
        # box_regression:(bs, num_anchors *  L, 2 *(1+num_classes))
        class_logits, box_regression = concat_box_prediction_layers(class_logits, box_regression, self.num_class)
        if self.training:
            """
            according to the ratio, that is (0.5, 0.5), to choose the positive and negative samples
            proposals: (7*L,2), represent index of  all the base anchors
            targets: out targets
            class_logits: (bs,len(selected_samples),(1+num_classes)), predicted label list
            box_regression:(bs, len(selected_samples),2* (1+num_classes))
            """
            proposals, matched_idxs, labels, regression_targets, class_logits, box_regression = \
                self.select_training_samples(proposals, targets, class_logits, box_regression)
        else:
            labels = None
            regression_targets = None

        # cat the class_logits, box_regression of different images to combine the batch
        # class_logits:(bs * num_anchors * L ,(1+num_classes)) if not training
        #              (bs * len(selected_samples), (1+num_classes)) otherwise
        class_logits, box_regression = self.cat_res(class_logits, box_regression)

        if self.training:
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            return loss_classifier + loss_box_reg
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shape)
            num_images = len(boxes)
            result = []
            for i in range(num_images):
                result.append({"boxes": boxes[i], "labels": labels[i], "scores": scores[i], })
            return result
