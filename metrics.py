from torch import Tensor


def cal_metrics(
    gt_clzes: list[Tensor],
    gt_poses: list[Tensor],
    pred_clzes: list[Tensor],
    pred_poses: list[Tensor],
    thresh: int = 3,
):
    """
    Calculate tp, fp and fn from targets and preds.

    Args:
        gt_clzes: n_sample n_tgt_event, the target classes of events belong to each sample.
        gt_poses: n_sample n_tgt_event, the target positiones of events belong to each sample.
        pred_clzes: n_sample n_pred_event, the pred classes of events belong to each sample.
        pred_poses: n_sample n_pred_event, the pred positiones of events belong to each sample.
        powers_list: n_sample L, the power sequence of each sample.
        is_record:
    """
    tp, fp, fn = 0, 0, 0
    ON, OFF = 1, 2
    for (
        gt_clzes_sample,
        gt_poses_sample,
        pred_clzes_sample,
        pred_poses_sample,
    ) in zip(gt_clzes, gt_poses, pred_clzes, pred_poses):
        tp_sample, fp_sample, fn_sample = 0, 0, 0
        for clz in (ON, OFF):
            gt_poses_sample_clz = gt_poses_sample[gt_clzes_sample == clz]
            pred_poses_sample_clz = pred_poses_sample[pred_clzes_sample == clz]
            if len(gt_poses_sample_clz) == 0:
                fp_sample += len(pred_poses_sample_clz)
            elif len(pred_poses_sample_clz) == 0:
                fn_sample += len(gt_poses_sample_clz)
            else:
                distanes = (
                    gt_poses_sample_clz[:, None] - pred_poses_sample_clz[None, :]
                ).abs()
                matched = []
                for distanes_ in distanes:
                    # distanes_ denote to the distanes of each pred events and current gt event
                    distanes_[matched] = thresh + 1
                    if distanes_.min().item() <= thresh:
                        matched.append(distanes_.argmin().item())
                tp_sample += len(matched)
                fp_sample += len(pred_poses_sample_clz) - len(matched)
                fn_sample += len(gt_poses_sample_clz) - len(matched)
        tp += tp_sample
        fp += fp_sample
        fn += fn_sample
    pre = tp / (tp + fp) if tp + fp > 0 else 0
    rec = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2.0 * pre * rec / (pre + rec) if pre + rec > 0 else 0
    return tp, fp, fn, pre, rec, f1
