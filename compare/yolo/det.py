"""
det.py provides the tools of detection
"""

import torch
import math
from typing import List, Tuple
from torch import Tensor


def box_iou(boxes1, boxes2):
    """ get iou of each pair in (boxes1 x boxes2), while the iou's 2 dims represent the box idx of boxes1 and boxes2 """
    # get length of all boxes
    area1 = boxes1[:, 1] - boxes1[:, 0]
    area2 = boxes2[:, 1] - boxes2[:, 0]
    # for each pair of (boxes1 x boxes2), get its larger left edge and lower right edge
    lt = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    rb = torch.min(boxes1[:, None, 1], boxes2[:, 1])
    # get intersection
    inter = (rb - lt).clamp(min=0)
    # calculate iou
    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def nms(boxes, scores, iou_threshold):
    """ return the idx of select boxes by nms """
    if boxes.shape[0] == 0:
        return np.array([]), np.array([])
    idx = np.argsort(scores.detach().cpu())
    pick = []
    while len(idx) > 0:
        pick.extend([idx[len(idx) - 1]])
        iou_mat = box_iou(boxes[idx[len(idx) - 1]].reshape(-1, 2), boxes[idx[:(len(idx) - 1)]])
        idx = np.delete(idx, np.concatenate(([len(idx) - 1], np.where(iou_mat.cpu().squeeze() > iou_threshold)[0])))
    pick = torch.as_tensor(pick, device=scores.device)
    return pick


def batched_nms(boxes, scores, labels, iou_threshold):
    """ based on nms strategy,filter boxes from the whole batch and return the idx of select boxes"""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    # add a large offset that is same of each class to all the boxes for performing nms of each class independently
    offsets = labels.to(boxes) * (boxes.max() + 1)
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


def remove_small_boxes(boxes, min_size):
    """ return the idx of boxes whose length not lower than the min_size """
    ws = boxes[:, 1] - boxes[:, 0]
    keep = torch.ge(ws, min_size)
    return torch.where(keep)[0]


def clip_boxes_to_image(boxes, size):
    """ Clip boxes so that they lie inside [0, size) """
    boxes_x = boxes[..., :]  # x1, x2
    return boxes_x.clamp(min=0, max=size - 1)  # 限制x坐标范围在[0,width]之间


class BalancedPositiveNegativeSampler(object):
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, batch_size_per_image, positive_fraction):
        """
        Arguments:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentage of positive elements per batch
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs):
        """
        return 2 masks of positive and negative respectively, where the mask of shape (bs, n_select_box)
        matched_idxs: shape of (bs, n_box)
        """
        pos_idx = []
        neg_idx = []

        for matched_idxs_per_image in matched_idxs:
            # get index of positive and negative boxes
            positive = torch.where(torch.ge(matched_idxs_per_image, 1))[0]
            negative = torch.where(torch.eq(matched_idxs_per_image, 0))[0]
            # calculate the num of positive and negative boxes
            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            num_neg = min(negative.numel(), num_neg)
            # randomly select positive and negative examples
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]
            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]
            # create binary mask from indices
            pos_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)
            neg_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)
            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1
            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)
        return pos_idx, neg_idx


def encode_boxes(reference_boxes, proposals, weights):
    """
    return center and width differences between proposal and gt_boxes
    $$
    diff(width) = wc * log(gt_width / pred_width)
    diff(center) = ww * (gt_center - pred_center) / pred_width
    $$
    where wc and ww denote to the weights of center and width differences,
          gt_center and gt_width denotes to center and width of gt_box,
          pred_center and pred_width denotes to center and width of pred box
    Arguments:
        reference_boxes (Tensor): reference boxes(gt)
        proposals (Tensor): boxes to be encoded(anchors)
        weights:
    """
    # unpack
    wx = weights[0]
    ww = weights[1]
    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_x2 = proposals[:, 1].unsqueeze(1)
    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 1].unsqueeze(1)
    # calculate width and center of proposals and gt_boxes respectively
    ex_widths = proposals_x2 - proposals_x1
    ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    # calculate differences of center and width between proposal and gt_boxes
    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets = torch.cat((targets_dx, targets_dw), dim=1)
    return targets


class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into the representation used for training the regressors.
    """

    def __init__(self, weights, bbox_xform_clip=math.log(5)):
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip  # 防止参数爆炸

    def encode(self, reference_boxes, proposals):
        """
        Args:
            reference_boxes: List[Tensor] gt_boxes
            proposals: List[Tensor] anchors
        Returns: regression parameters
        """
        # record boxes num of all sample in batch and then concat gt_boxes and proposals respectively
        boxes_per_image = [len(b) for b in reference_boxes]
        reference_boxes = torch.cat(reference_boxes, dim=0)
        proposals = torch.cat(proposals, dim=0)
        # get differences of width and center between gt_boxes and proposals
        targets = self.encode_single(reference_boxes, proposals)
        # split result through recorded boxes num per sample
        return targets.split(boxes_per_image)

    def encode_single(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some reference boxes
        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        targets = encode_boxes(reference_boxes, proposals, weights)
        return targets

    def decode(self, rel_codes, boxes):
        """
        [模型预测的中心，宽度偏移]，【铺了模板之后的左右偏移】
        Args:
            rel_codes: bbox regression parameters
            boxes: anchors
        """
        assert isinstance(boxes, (list, tuple))
        assert isinstance(rel_codes, torch.Tensor)
        boxes_per_image = [b.size(0) for b in boxes]
        concat_boxes = torch.cat(boxes, dim=0)
        box_sum = 0
        for val in boxes_per_image:
            box_sum += val
        # get left and right bounds of boxes
        pred_boxes = self.decode_single(rel_codes.reshape(box_sum, -1), concat_boxes)
        return pred_boxes.reshape(box_sum, -1, 2)

    def decode_single(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets, get the decoded boxes.
        Arguments:
            rel_codes (Tensor): encoded boxes (bbox regression parameters)
            boxes (Tensor): reference boxes (anchors)
        """
        boxes = boxes.to(rel_codes.dtype)

        # get width and center of boxes
        widths = boxes[:, 1] - boxes[:, 0]
        ctr_x = boxes[:, 0] + 0.5 * widths
        # get width and center differences as dw and dx
        wx, ww = self.weights  # 默认都是1
        dx = rel_codes[:, 0::2] / wx
        dw = rel_codes[:, 1::2] / ww

        # limit max value, prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        # calculate pred width and center of box as pred_w and pred_ctr_x
        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_w = torch.exp(dw) * widths[:, None]

        # calculate left and right bounds of boxes
        pred_boxes1 = pred_ctr_x - torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        pred_boxes3 = pred_ctr_x + torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        pred_boxes = torch.stack((pred_boxes1, pred_boxes3), dim=2).flatten(1)
        return pred_boxes


class Matcher(object):
    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        """
        Args:
            high_threshold (float): quality values greater than or equal to
                this value are candidate matches.
            low_threshold (float): a lower quality threshold used to stratify
                matches into three levels:
                1) matches >= high_threshold
                2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
                3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions that have only low-quality match candidates. See
                set_low_quality_matches_ for more details.
        """
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold  # 0.7
        self.low_threshold = low_threshold  # 0.3
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        计算anchors与每个gtboxes匹配的iou最大值，并记录索引，
        iou<low_threshold索引值为-1， low_threshold<=iou<high_threshold索引值为-2
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
            pairwise quality between M ground-truth elements and N predicted elements.

        Returns:
            matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
            [0, M - 1] or a negative value indicating that prediction i could not
            be matched.
        """
        if match_quality_matrix.numel() == 0:
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    "No ground-truth boxes available for one of the images "
                    "during training")
            else:
                raise ValueError(
                    "No proposal boxes available for one of the images "
                    "during training")

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        # M x N 的每一列代表一个anchors与所有gt的匹配iou值
        # matched_vals代表每列的最大值，即每个anchors与所有gt匹配的最大iou值
        # matches对应最大值所在的索引
        matched_vals, matches = match_quality_matrix.max(dim=0)  # the dimension to reduce.
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        else:
            all_matches = None

        # Assign candidate matches with low quality to negative (unassigned) values
        # 计算iou小于low_threshold的索引
        below_low_threshold = matched_vals < self.low_threshold
        # 计算iou在low_threshold与high_threshold之间的索引值
        between_thresholds = (matched_vals >= self.low_threshold) & (matched_vals <= self.high_threshold)
        # iou小于low_threshold的matches索引置为-1
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD  # -1
        # iou在[low_threshold, high_threshold]之间的matches索引置为-2
        matches[between_thresholds] = self.BETWEEN_THRESHOLDS  # -2

        if self.allow_low_quality_matches:
            assert all_matches is not None
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches

    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth with which it has the highest
        quality value.
        """
        # For each gt, find the prediction with which it has highest quality
        # 对于每个gt boxes寻找与其iou最大的anchor，
        # highest_quality_foreach_gt为匹配到的最大iou值
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)  # the dimension to reduce.

        # Find highest quality match available, even if it is low, including ties
        # 寻找每个gt boxes与其iou最大的anchor索引，一个gt匹配到的最大iou可能有多个anchor
        # gt_pred_pairs_of_highest_quality = torch.nonzero(
        #     match_quality_matrix == highest_quality_foreach_gt[:, None]
        # )
        gt_pred_pairs_of_highest_quality = torch.where(
            torch.eq(match_quality_matrix, highest_quality_foreach_gt[:, None])
        )
        # gt_pred_pairs_of_highest_quality[:, 0]代表是对应的gt index(不需要)
        # pre_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
        pre_inds_to_update = gt_pred_pairs_of_highest_quality[1]
        # 保留该anchor匹配gt最大iou的索引，即使iou低于设定的阈值
        matches[pre_inds_to_update] = all_matches[pre_inds_to_update]


def smooth_l1_loss(input, target, beta: float = 1. / 9, size_average: bool = True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    # cond = n < beta
    cond = torch.lt(n, beta)
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()


##########################################################################################
"""
eval_tools
"""
from collections import defaultdict
import numpy as np
import pandas as pd
import os


def eval_detection_voc(ids, imglist,
                       pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
                       gt_difficults=None,
                       iou_thresh=0.5, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.

    This function evaluates predicted bounding boxes obtained from a dataset
    which has :math:`N` images by using average precision for each class.
    The code is based on the evaluation code used in PASCAL VOC Challenge.

    Args:
        pred_bboxes (iterable of numpy.ndarray): An iterable of :math:`N`
            sets of bounding boxes.
            Its index corresponds to an index for the base dataset.
            Each element of :obj:`pred_bboxes` is a set of coordinates
            of bounding boxes. This is an array whose shape is :math:`(R, 2)`,
            where :math:`R` corresponds
            to the number of bounding boxes, which may vary among boxes.
            The second axis corresponds to
            :math:`x_{min}, x_{max}` of a bounding box.
        pred_labels (iterable of numpy.ndarray): An iterable of labels.
            Similar to :obj:`pred_bboxes`, its index corresponds to an
            index for the base dataset. Its length is :math:`N`.
        pred_scores (iterable of numpy.ndarray): An iterable of confidence
            scores for predicted bounding boxes. Similar to :obj:`pred_bboxes`,
            its index corresponds to an index for the base dataset.
            Its length is :math:`N`.
        gt_bboxes (iterable of numpy.ndarray): An iterable of ground truth
            bounding boxes
            whose length is :math:`N`. An element of :obj:`gt_bboxes` is a
            bounding box whose shape is :math:`(R, 4)`. Note that the number of
            bounding boxes in each image does not need to be same as the number
            of corresponding predicted boxes.
        gt_labels (iterable of numpy.ndarray): An iterable of ground truth
            labels which are organized similarly to :obj:`gt_bboxes`.
        gt_difficults (iterable of numpy.ndarray): An iterable of boolean
            arrays which is organized similarly to :obj:`gt_bboxes`.
            This tells whether the
            corresponding ground truth bounding box is difficult or not.
            By default, this is :obj:`None`. In that case, this function
            considers all bounding boxes to be not difficult.
        iou_thresh (float): A prediction is correct if its Intersection over
            Union with the ground truth is above this value.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.

    Returns:
        dict:

        The keys, value-types and the description of the values are listed
        below.

        * **ap** (*numpy.ndarray*): An array of average precisions. \
            The :math:`l`-th value corresponds to the average precision \
            for class :math:`l`. If class :math:`l` does not exist in \
            either :obj:`pred_labels` or :obj:`gt_labels`, the corresponding \
            value is set to :obj:`numpy.nan`.
        * **map** (*float*): The average of Average Precisions over classes.

    """

    precsum, recsum, prec, rec = calc_detection_voc_prec_rec(ids, imglist,
                                                             pred_bboxes, pred_labels, pred_scores,
                                                             gt_bboxes, gt_labels, gt_difficults,
                                                             iou_thresh=iou_thresh)

    # ap = calc_detection_voc_ap(prec, rec, use_07_metric=use_07_metric)

    # return {'ap': ap, 'map': np.nanmean(ap)}
    return {'prec': precsum, 'rec': recsum}


def getious(pred_bbox, pred_label, gt_bbox, gt_label):  # 求LA要用的，计算分类对且与真实框的交并比在0.5以上的预测框的iou的值
    if pred_bbox.shape[0] == 0 or gt_bbox.shape[0] == 0:
        return np.array([])
    pred_bbox = pred_bbox.round()
    iou = box_iou(pred_bbox, gt_bbox)
    gt_index = iou.argmax(axis=1)
    right = np.where(pred_label == gt_label[gt_index])[0]
    iou = iou[right, gt_index[right]]
    ious_index = np.where(iou >= 0.5)[0]
    ious = iou[ious_index]
    return ious


def calc_detection_voc_prec_rec(ids,
                                pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, args, epoch,
                                seemode=True,
                                iou_thresh=0.5, flagg=False):
    """Calculate precision and recall based on evaluation code of PASCAL VOC.

    This function calculates precision and recall of
    predicted bounding boxes obtained from a dataset which has :math:`N`
    images.
    The code is based on the evaluation code used in PASCAL VOC Challenge.

    Args:
        pred_bboxes (iterable of numpy.ndarray): An iterable of :math:`N`
            sets of bounding boxes.
            Its index corresponds to an index for the base dataset.
            Each element of :obj:`pred_bboxes` is a set of coordinates
            of bounding boxes. This is an array whose shape is :math:`(R, 2)`,
            where :math:`R` corresponds
            to the number of bounding boxes, which may vary among boxes.
            The second axis corresponds to
            :math:` x_{min}, x_{max}` of a bounding box.
        pred_labels (iterable of numpy.ndarray): An iterable of labels.
            Similar to :obj:`pred_bboxes`, its index corresponds to an
            index for the base dataset. Its length is :math:`N`.
        pred_scores (iterable of numpy.ndarray): An iterable of confidence
            scores for predicted bounding boxes. Similar to :obj:`pred_bboxes`,
            its index corresponds to an index for the base dataset.
            Its length is :math:`N`.
        gt_bboxes (iterable of numpy.ndarray): An iterable of ground truth
            bounding boxes
            whose length is :math:`N`. An element of :obj:`gt_bboxes` is a
            bounding box whose shape is :math:`(R, 2)`. Note that the number of
            bounding boxes in each image does not need to be same as the number
            of corresponding predicted boxes.
        gt_labels (iterable of numpy.ndarray): An iterable of ground truth
            labels which are organized similarly to :obj:`gt_bboxes`.
        gt_difficults (iterable of numpy.ndarray): An iterable of boolean
            arrays which is organized similarly to :obj:`gt_bboxes`.
            This tells whether the
            corresponding ground truth bounding box is difficult or not.
            By default, this is :obj:`None`. In that case, this function
            considers all bounding boxes to be not difficult.
        iou_thresh (float): A prediction is correct if its Intersection over
            Union with the ground truth is above this value..

    Returns:
        tuple of two lists:
        This function returns two lists: :obj:`prec` and :obj:`rec`.

        * :obj:`prec`: A list of arrays. :obj:`prec[l]` is precision \
            for class :math:`l`. If class :math:`l` does not exist in \
            either :obj:`pred_labels` or :obj:`gt_labels`, :obj:`prec[l]` is \
            set to :obj:`None`.
        * :obj:`rec`: A list of arrays. :obj:`rec[l]` is recall \
            for class :math:`l`. If class :math:`l` that is not marked as \
            difficult does not exist in \
            :obj:`gt_labels`, :obj:`rec[l]` is \
            set to :obj:`None`.

    """

    pred_bboxes = iter(pred_bboxes)
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)
    gt_bboxes = iter(gt_bboxes)
    gt_labels = iter(gt_labels)

    n_pos = defaultdict(int)  # n_pos[i]表示第i类事件共有多少个非困难事件
    score = defaultdict(list)  # score[i]是一个列表，表示第i类事件所有预测为第i类事件的得分
    match = defaultdict(list)  # match[i]是一个列表，表示所有被预测为i类的事件是否正确 1正确 0错误
    tpsum = 0  # TP
    fpsum = 0  # FP
    n_pos_sum = 0  # 总目标数
    ioulist = []
    ssp = []  # 存储错误数据及其错误类型
    # 1：有目标但没预测出来【开始时间，真实框，真实类别，0,1】
    # 2：没目标瞎预测【开始时间，预测框，预测类别，预测分数，2】
    # 3：#重复预测目标了，感觉不太会出现，两个回归框对应同一个gt,且两个框的交并比还小，没有被nms掉【开始时间，预测框，预测类别，预测分数，3】
    # 4：一个时间窗有多个同类事件，但是全部预测到简单识别的那个了，把另一个漏了【开始时间，真实框，真实类别，0,4】
    for id, pred_bbox, pred_label, pred_score, gt_bbox, gt_label in zip(ids,
                                                                        pred_bboxes, pred_labels, pred_scores,
                                                                        gt_bboxes, gt_labels):

        # ---以下计算LA location accuracy 所需的所有正确的pred_bbox与其gtbox的iou：
        ioulist.append(getious(pred_bbox, pred_label, gt_bbox, gt_label))
        # print(pred_bbox)

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):  # l遍历所有可能出现的种类
            pred_mask_l = pred_label == l  # 注意后面是‘==’
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            pred_label_l = pred_label[pred_mask_l]
            # sort by score
            order = np.array(pred_score_l.argsort())[::-1].copy()
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]

            n_pos[l] += gt_mask_l.sum()  # 计算真实情况下每类的个数

            score[l].extend(pred_score_l)  # 统计预测的每类的分数

            if len(pred_bbox_l) == 0 and len(gt_bbox_l) != 0:  # 有目标但没预测出来
                if seemode:
                    # print('---miss---1')
                    # print(id)#start time
                    p = []
                    for box0 in gt_bbox_l:
                        # print(box0)
                        # print(l)
                        # p.append([id.numpy(),box0.numpy(),l.numpy(),0,1])
                        ssp.append([id.item(), box0[0].item(), box0[1].item(), l, 0, 1])
                continue
            if len(gt_bbox_l) == 0 and len(pred_bbox_l) != 0:  # 没目标但预测出来了
                match[l].extend((0,) * pred_bbox_l.shape[0])
                if seemode:
                    dd = []
                    for box0, label0, score0 in zip(pred_bbox_l, pred_label_l, pred_score_l):
                        ssp.append([id.item(), box0[0].item(), box0[1].item(), label0.item(), score0.item(), 2])
                continue

            # VOC evaluation follows integer typed bounding boxes.
            pred_bbox_l = pred_bbox_l
            pred_bbox_l[:, 1:] += 1
            gt_bbox_l = gt_bbox_l
            gt_bbox_l[:, 1:] += 1

            iou = box_iou(pred_bbox_l, gt_bbox_l)
            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            gt_index[np.array(iou.max(axis=1)[0]) < iou_thresh] = -1  # 交并比小于阈值的不参与计数
            del iou

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)  # 是否已经有其他pred_box 是对应这个gt_bbox的标记

            for i, gt_idx in enumerate(gt_index):
                if gt_idx >= 0:
                    if not selec[gt_idx]:
                        match[l].append(1)  # 一个gt_box如果有多个pred_box对应到它，只有第一个能算对
                    else:
                        match[l].append(0)
                        if seemode:
                            # print('---repeate---')#重复预测目标了，感觉不太会出现，两个回归框对应同一个gt,且两个框的交并比还小，没有被nms掉
                            # print(id)
                            # print(pred_bbox_l[i])
                            # print(pred_label_l[i])
                            # print(pred_score_l[i])
                            ssp.append(
                                [id.item(), pred_bbox_l[i][0].item(), pred_bbox_l[i][1].item(), pred_label_l[i].item(),
                                 pred_score_l[i].item(), 3])

                    selec[gt_idx] = True
                else:
                    match[l].append(-1)
            if seemode:
                for i, select in enumerate(selec):
                    if not select:
                        # print('---miss---2')#有多个同类事件，但是全部预测到简单识别的那个了
                        # print(id)
                        # print(gt_bbox_l[i])
                        # print(l)
                        ssp.append([id.item(), gt_bbox_l[i][0].item(), gt_bbox_l[i][1].item(), l, 0, 4])
    for iter_ in (
            pred_bboxes, pred_labels, pred_scores,
            gt_bboxes, gt_labels):
        if next(iter_,
                None) is not None:  # 这些pred_bboxes, pred_labels, pred_scores,gt_bboxes, gt_labels, gt_difficults都已经被转换成iter对象在for循环里面被访问了，如果还有next不是None的，说明长度不一样，报错
            raise ValueError('Length of input iterables need to be same.')

    n_fg_class = max(n_pos.keys()) + 1  # 有多少类别
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    if not os.path.exists(args.false_data + str(args.cla_shebei)):
        os.makedirs(args.false_data + str(args.cla_shebei))
    ssp = pd.DataFrame(ssp)
    if (flagg == True):
        # ssp.to_csv(args.false_data+str(args.cla_shebei)+'/'+'cuowushuju_%s.csv'%(str(epoch)), sep=" ", header=None, index=None)
        ssp.to_csv(args.false_data + 'falsedd' + str(args.cla_shebei) + '/' + 'cuowushuju_%s.csv' % (str(epoch)),
                   sep=" ", header=None, index=None)

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1].copy()
        match_l = match_l[order]

        tp = np.sum(match_l == 1)
        fp = np.sum(match_l == 0)

        tpsum += tp
        fpsum += fp
        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]
            n_pos_sum += n_pos[l]
    precsum = tpsum / (tpsum + fpsum)
    recsum = tpsum / n_pos_sum
    if True:
        print('各类的数量', n_pos)
        print('各类的精度：', prec)
        print('各类的召回率：', rec)
        print('精度：' + str(precsum))
        print('召回率：' + str(recsum))
        print('f1_score:' + str(2 * precsum * recsum / (precsum + recsum)))
        print('事件总数：' + str(n_pos_sum))
    # ---以下计算LA location accuracy
    ioulist = np.concatenate(ioulist)
    print('LA:预测对的框的iou均值' + str(ioulist.mean()))
    print('LA_std:方差' + str(ioulist.std()))
    return 2 * precsum * recsum / (precsum + recsum), precsum, recsum
