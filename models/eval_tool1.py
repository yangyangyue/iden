
from __future__ import division
from collections import defaultdict
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import os
from yolo.det import box_iou
def read_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    box_list = []
    class_list=[]
    type_=[]
    for child_of_root in root:
        if child_of_root.tag == 'object':
            label = None
            for child_item in child_of_root:
                if child_item.tag == 'name':
                    label = int(child_item.text.split('-')[-1])
                    # label = 1#构造波动数据集，则只需要对所有有事件的目标进行标签为1的设置


                if child_item.tag == 'bndbox':
                    tmp_box = []
                    for node in child_item:
                        tmp_box.append(int(node.text))
                    assert label is not None, 'label is none, error'
                    class_list.append(label)
                    box_list.append(tmp_box)
                if child_item.tag == 'difficult':
                    type_.append(int(child_item.text))


    return class_list,box_list,type_


def eval_detection_voc(ids, imglist,
        pred_bboxes, pred_labels, gt_bboxes, gt_labels,
        gt_difficults=None,
        iou_thresh=0.5):
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
        pred_bboxes, pred_labels,
        gt_bboxes, gt_labels, gt_difficults,
        iou_thresh=iou_thresh)

    # ap = calc_detection_voc_ap(prec, rec, use_07_metric=use_07_metric)

    # return {'ap': ap, 'map': np.nanmean(ap)}
    return {'prec':precsum,'rec':recsum}


def getious(pred_bbox,pred_label,gt_bbox,gt_label):  # 求LA要用的，计算分类对且与真实框的交并比在0.5以上的预测框的iou的值
    if pred_bbox.shape[0] == 0 or gt_bbox.shape[0] == 0:
        return np.array([])
    # pred_bbox = pred_bbox.round()
    iou = box_iou(pred_bbox, gt_bbox)
    gt_index = iou.argmax(axis=1)
    right = np.where(pred_label == gt_label[gt_index])[0]
    iou = iou[right,gt_index[right]]
    ious_index = np.where(iou>=0.5)[0]
    ious = iou[ious_index]
    return ious


def calc_detection_voc_prec_rec(ids,
        pred_bboxes, pred_labels, gt_bboxes, gt_labels,args,
        seemode=True, iou_thresh=0.03):

    interval_qujian=np.loadtxt(args.false_data+'ImageSets/interval.txt')
    if args.name=='redd':
        if args.cla_shebei==10:
            gonglv=[[194.04000000000008, 485.74, 437.24, 1360.74, 148.33000000000004, 670.5600000000001],
                    [102.6099999999999, 353.02, 396.08, 1148.7200000000003, 107.90000000000009, 600.31]]
        if args.cla_shebei==14:
            gonglv=[[3204.4799999999996, 2963.92, 2487.9500000000003, 511.35],[2845.33, 2317.26, 2042.58, 406.8399999999999]]
        if args.cla_shebei==16:
            gonglv=[[1957.1799999999998, 1882.19, 165.42000000000002, 10000],
                    [1587.6400000000003, 1573.4299999999998, 100, 29.570000000000007]]

    if args.name=='ukdale':
        if args.cla_shebei==8:
            gonglv=[[3200,3200],[2800,2700]]
        if args.cla_shebei==9:
            gonglv=[[450,400],[350,340]]
        if args.cla_shebei==13:
            gonglv=[[2186.4799999999996, 1966, 135,129],[1913.33, 1828, 84, 93]]
        if args.cla_shebei==15:
            gonglv=[[1400,1450],[1100,1100]]

    if args.name=='redd':
        if args.cla_shebei==10:
            gonglv_wending=[[194.04000000000008, 485.74, 437.24, 1360.74, 148.33000000000004, 670.5600000000001],
                    [102.6099999999999, 353.02, 396.08, 1148.7200000000003, 107.90000000000009, 600.31]]
        if args.cla_shebei==14:
            gonglv_wending=[[3204.4799999999996, 2963.92, 2487.9500000000003, 511.35],[2845.33, 2317.26, 2042.58, 406.8399999999999]]
        if args.cla_shebei==16:
            gonglv_wending=[[1957.1799999999998, 1882.19, 165.42000000000002, 54],
                    [1587.6400000000003, 1573.4299999999998, 100, 29.570000000000007]]
    if args.name=='ukdale':
        if args.cla_shebei==8:
            gonglv_wending=[[3200,3200],[2800,2700]]
        if args.cla_shebei==9:
            gonglv_wending=[[42.5,25],[42.5,25]]
        if args.cla_shebei==13:
            gonglv_wending=[[2186.4799999999996, 2030, 135,129],[1913.33, 1828, 84, 93]]
        if args.cla_shebei==15:
            gonglv_wending=[[400,400],[400,400]]
    device_list=[9,15]

    summ = 0
    ignore= 0

    pred_bboxes = iter(pred_bboxes)
    pred_labels = iter(pred_labels)
    gt_bboxes = iter(gt_bboxes)
    gt_labels = iter(gt_labels)

    n_pos = defaultdict(int)  # n_pos[i]表示第i类事件共有多少个非困难事件
    match = defaultdict(list)  # match[i]是一个列表，表示所有被预测为i类的事件是否正确 1正确 0错误
    tpsum = 0  # TP
    fpsum= 0  # FP
    n_pos_sum = 0  # 总目标数
    ioulist = []
    ssp = []    #存储错误数据及其错误类型
    true_result=[]#存储正确预测的数据方便后续的输出和使用
    #1：有目标但没预测出来【开始时间，真实框，真实类别，0,0,0,0,1】
    #2：没目标瞎预测【开始时间，预测框，预测类别，预测分数，功率幅值，前功率区间，后功率区间，2】
    #3：#重复预测目标了，感觉不太会出现，两个回归框对应同一个gt,且两个框的交并比还小，没有被nms掉【开始时间，预测框，预测类别，预测分数，功率幅值，前功率区间，后功率区间，3】
    #4：一个时间窗有多个事件，但是漏了个【开始时间，真实框，真实类别，0,0,0,0，4】
    #5：一个时间窗预测事件，预测错误的框【开始时间，预测框，预测类别，预测分数，功率幅值，前功率区间，后功率区间，5】,例如实际一个，预测了两个
    #6:因为不在功率区间或者功率不稳定造成的错误，为了保持和上述的后处理方式一样，对正确的也做了后处理
    for id,pred_bbox, pred_label, gt_bbox, gt_label in zip(ids, pred_bboxes, pred_labels, gt_bboxes, gt_labels):

        # ---以下计算LA location accuracy 所需的所有正确的pred_bbox与其gtbox的iou：
        ioulist.append(getious(pred_bbox,pred_label,gt_bbox,gt_label))
        data = np.loadtxt(args.false_data + 'JPEGImages/%d.txt' % (id))
        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):  # l遍历所有可能出现的种类
            pred_mask_l = pred_label == l  # 注意后面是‘==’
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_label_l = pred_label[pred_mask_l]
            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            n_pos[l] += gt_mask_l.sum()  #计算真实情况下每类的个数


            if len(pred_bbox_l) == 0 and len(gt_bbox_l)!=0:#有目标但没预测出来
                if seemode:
                    # print('---miss---1')
                    # print(id)#start time
                    p=[]
                    for box0 in gt_bbox_l:
                        ssp.append([id.item(), box0[0].item(), box0[1].item(), l, 0,0,0,0, 1])
                continue

            if len(gt_bbox_l) == 0 and len(pred_bbox_l) != 0:#没目标但预测出来了
                for box0,label0 in zip(pred_bbox_l,pred_label_l):

                    fl_box0=box0[0].item()#预测框左边界
                    fl_box1=box0[1].item()#预测框右边界
                    box0_0 = int(max(0, round(box0[0].item())))  # 化整后的左边界
                    box0_1 = int(min(1023, round(box0[1].item())))  # 化整后的右边界
                    # box0_0=int(max(0,box0[0].item()))#化整后的左边界
                    # box0_1=int(min(1023,box0[1].item()))#化整后的右边界
                    #计算是否在断裂区间，在的话不算错误
                    flag_duanlie = 0
                    for k in interval_qujian:
                        if(max(data[box0_0,0],k[0])<min(data[box0_1,0],k[1])):
                            flag_duanlie=1
                            break
                    #计算是功率是否在区间内，不在功率区间内的话也不算错误，因为会被扔出去
                    flag_gonglv=0
                    gonglv_=max(data[box0_0:(box0_1+1), 1])-min(data[box0_0:(box0_1+1), 1])#预测区间的功率幅值
                    if (gonglv_<=gonglv[0][l-1])and (gonglv_>=gonglv[1][l-1]):
                        flag_gonglv=1#在功率区间内

                    if (args.cla_shebei == 16 and l == 4 and abs(data[box0_0, 1] - data[(box0_1), 1]) >= 30):
                        flag_gonglv = 0

                    flag_qushiwending = 0  # 判断趋势是否稳定
                    if (box0_1 - box0_0 >= 1):

                        qujian_front = abs(data[box0_0, 1] - data[(box0_0 + 1), 1])  # 区间的前半部分稳定功率
                        qujian_end = abs(data[box0_1, 1] - data[(box0_1 - 1), 1])  # 区间的后半部分稳定功率
                    else:
                        qujian_front = 0  # 区间的前半部分稳定功率
                        qujian_end = 0  # 区间的后半部分稳定功率

                    if (args.cla_shebei in device_list):
                        box0_0 = int(max(0, box0[0].item()))  # 化整后的左边界
                        box0_1 = int(min(1023, box0[1].item()))  # 化整后的右边界
                        qujian_front = abs(data[box0_0, 1] - data[(box0_0 + 1), 1])  # 区间的前半部分稳定功率
                        qujian_end = abs(data[box0_1, 1] - data[(box0_1 - 1), 1])  # 区间的后半部分稳定功率
                        if (box0_1 - box0_0 >= 1) and (qujian_front <= gonglv_wending[0][l-1]) and (qujian_end <= gonglv_wending[1][l-1]):
                            flag_qushiwending = 1
                    else:
                        flag_qushiwending = 1

                    if(flag_duanlie==0 and flag_gonglv==1 and flag_qushiwending==1):
                        summ += 1
                        match[l].append(0)
                        ssp.append([id.item(), fl_box0, fl_box1, label0.item(),gonglv_,qujian_front,qujian_end, 2])

                continue

            iou = box_iou(pred_bbox_l, gt_bbox_l)
            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            gt_index[np.array(iou.max(axis=1)[0]) < iou_thresh] = -1#交并比小于阈值的不参与计数
            del iou

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)  # 是否已经有其他pred_box 是对应这个gt_bbox的标记

            for i,gt_idx in enumerate(gt_index):
                if gt_idx >= 0:
                    if not selec[gt_idx]:
                        box0 = pred_bbox_l[i]
                        fl_box0 = box0[0].item()  # 预测框左边界
                        fl_box1 = box0[1].item()#-1  # 预测框右边界
                        box0_0 = int(max(0, round(box0[0].item())))  # 化整后的左边界
                        box0_1 = int(min(1023, round(box0[1].item())))  # 化整后的右边界

                        flag_duanlie = 0
                        for k in interval_qujian:
                            if (max(data[box0_0, 0], k[0]) < min(data[box0_1, 0], k[1])):
                                flag_duanlie = 1
                                break

                        # 计算是功率是否在区间内，不在功率区间内的话也不算错误，因为会被扔出去
                        flag_gonglv = 0
                        gonglv_ = max(data[box0_0:(box0_1 + 1), 1]) - min(data[box0_0:(box0_1 + 1), 1])  # 预测区间的功率幅值
                        if (gonglv_ <= gonglv[0][l - 1]) and (gonglv_ >= gonglv[1][l - 1]):
                            flag_gonglv = 1  # 在功率区间内

                        # redd16号用电器的四事件是一个凸起，始末功率接近
                        if (args.cla_shebei == 16 and l == 4 and abs(data[box0_0, 1] - data[(box0_1), 1]) >= 30):
                            flag_gonglv = 0

                        flag_qushiwending = 0  # 判断趋势是否稳定
                        if (box0_1 - box0_0 >= 1):
                            qujian_front = abs(data[box0_0, 1] - data[(box0_0 + 1), 1])  # 区间的前半部分稳定功率
                            qujian_end = abs(data[box0_1, 1] - data[(box0_1 - 1), 1])  # 区间的后半部分稳定功率
                        else:
                            qujian_front = 0  # 区间的前半部分稳定功率
                            qujian_end = 0  # 区间的后半部分稳定功率

                        if (args.cla_shebei in device_list):
                            if (box0_1 - box0_0 >= 1) and (qujian_front <= gonglv_wending[0][l-1]) and (qujian_end <= gonglv_wending[1][l-1]):
                                flag_qushiwending = 1
                        else:
                            flag_qushiwending = 1

                        if (flag_duanlie==0 and flag_gonglv == 1 and flag_qushiwending == 1):
                            selec[gt_idx] = True
                            summ += 1
                            match[l].append(1)
                            # print(id)
                            true_result.append([id.item(), fl_box0, fl_box1, l,gt_bbox_l[gt_idx][0].item(),gt_bbox_l[gt_idx][1].item()-1])
                        else:
                            if(args.cla_shebei == 16 and l == 4):
                                cd=2
                            ssp.append(
                                [id.item(), fl_box0, fl_box1, l, gonglv_, gt_bbox_l[gt_idx][0].item(),
                                 gt_bbox_l[gt_idx][1].item()-1, 6])####
                    else:
                        ignore += 1
                        summ += 1
                        # match[l].append(0)
                        if seemode:
                            # print('---repeate---')#重复预测目标了，感觉不太会出现，两个回归框对应同一个gt,且两个框的交并比还小，没有被nms掉
                            ssp.append([id.item(),pred_bbox_l[i][0].item(),pred_bbox_l[i][1].item(),pred_label_l[i].item(),0,0,0,3])


                else:
                    # match[l].append(0)
                    box0 = pred_bbox_l[i]
                    fl_box0 = pred_bbox_l[i][0].item()# 预测框左边界
                    fl_box1 = pred_bbox_l[i][1].item()#-1# 预测框右边界
                    # box0_0 = int(max(0, box0[0].item()))  # 化整后的左边界
                    # box0_1 = int(min(1023, box0[1].item()))  # 化整后的右边界
                    box0_0 = int(max(0, round(box0[0].item())))  # 化整后的左边界
                    box0_1 = int(min(1023, round(box0[1].item())))  # 化整后的右边界

                    # 计算是否在断裂区间，在的话不算错误
                    flag_duanlie = 0
                    for k in interval_qujian:
                        if (max(data[box0_0, 0], k[0]) < min(data[box0_1, 0], k[1])):
                            flag_duanlie = 1
                            break
                    # 计算是功率是否在区间内，不在功率区间内的话也不算错误，因为会被扔出去
                    flag_gonglv = 0
                    gonglv_ = max(data[box0_0:(box0_1 + 1), 1]) - min(data[box0_0:(box0_1 + 1), 1])  # 预测区间的功率幅值
                    if (gonglv_ <= gonglv[0][l - 1]) and (gonglv_ >= gonglv[1][l - 1]):
                        flag_gonglv = 1  # 在功率区间内

                        # redd16号用电器的四事件是一个凸起，始末功率接近
                    if (args.cla_shebei == 16 and l == 4 and abs(data[box0_0, 1] - data[(box0_1), 1]) >= 30):
                        flag_gonglv = 0

                    flag_qushiwending = 0  # 判断趋势是否稳定
                    if (box0_1 - box0_0 >= 1):
                        qujian_front = abs(data[box0_0, 1] - data[(box0_0 + 1), 1])  # 区间的前半部分稳定功率
                        qujian_end = abs(data[box0_1, 1] - data[(box0_1 - 1), 1])  # 区间的后半部分稳定功率
                    else:
                        qujian_front = 0  # 区间的前半部分稳定功率
                        qujian_end = 0  # 区间的后半部分稳定功率

                    if (args.cla_shebei in device_list):
                        if (box0_1 - box0_0 >= 1) and (qujian_front <= gonglv_wending[0][l-1]) and (qujian_end <= gonglv_wending[1][l-1]):
                            flag_qushiwending = 1
                    else:
                        flag_qushiwending = 1

                    if (flag_duanlie == 0 and flag_gonglv == 1 and flag_qushiwending == 1):
                        summ += 1
                        match[l].append(0)
                        ssp.append([id.item(), fl_box0, fl_box1, pred_label_l[i].item(), gonglv_, qujian_front,
                                    qujian_end, 5])

                    del box0,fl_box1,fl_box0,



            if seemode:
                for i,select in enumerate(selec):
                    if not select:
                        ssp.append([id.item(), gt_bbox_l[i][0].item(),gt_bbox_l[i][1].item()-1 ,l, 0,0,0,0,4])

    n_fg_class = max(n_pos.keys()) + 1#有多少类别
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    print(f"yours: summ-{summ}, ignore-{ignore}")

    for l in n_pos.keys():
        match_l = np.array(match[l], dtype=np.int8)

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
    precsum = tpsum / (tpsum+fpsum)
    recsum = tpsum / n_pos_sum
    if True:
        # print('各类的数量',n_pos)#数量不按顺序但是精度召回率按顺序
        # print('各类的精度：',prec)
        # print('各类的召回率：',rec)
        print(f"tp: {tpsum}, fp: {fpsum}, n_pos: {n_pos_sum}")
        print('f1_score:' + str(2 * precsum * recsum / (precsum + recsum)))
        print('精度：'+str(precsum))
        print('召回率：'+str(recsum))
        # print('f1_score:' + str(2*precsum*recsum/(precsum+recsum)))
        # print('事件总数：'+str(n_pos_sum))
        # print('tp', tpsum, 'fp', fpsum, 'fn', n_pos_sum - tpsum)
    # ---以下计算LA location accuracy
    ioulist = np.concatenate(ioulist)
    print('LA:预测对的框的iou均值' + str(ioulist.mean()))
    print('LA_std:方差' + str(ioulist.std()))
    return 2*precsum*recsum/(precsum+recsum),precsum,recsum