# Copyright 2023 lily. All rights reserved.
#
# Author: lily
# Email: lily231147@proton.me


from pathlib import Path
import shutil

import numpy as np

from dataset import read, threshs, sizes


def annotate(set_name: str, house_id: str, app_abb: str, search_length: int = 30) -> tuple:
    """
    对UKDALE总线(1s)上的事件时间戳进行标注。简而言之，首先通过预设阈值在支线上找到事件时间戳，然后在主线范围内匹配特定事件。
    对于每个事件标注总线时间戳、事件类别、事件类型和支线时间戳。事件类别指的是开启事件(1)和关闭事件(2)。事件类型匹配事件(1)和不匹配事件(2)。

    3种可能的标注错误(占比很小 忽略):
        负载信号上不存在事件，但主线上存在，这可能导致标签遗漏。
        负载信号上存在事件 但主线上未找到 这可能是由于事件重叠、负载信号与主线之间存在较大偏移或采集问题造成的。这些事件的类型被指定为1。
        在主线上匹配了错误的事件。
    """
    # 读取总线和支线，同时截取处于总线范围内的支线
    mains = read(set_name, house_id)
    apps = read(set_name, house_id, app_abb)
    begin = np.searchsorted(apps[:, 0], mains[0, 0])
    end = np.searchsorted(apps[:, 0], mains[-1, 0])
    apps = apps[begin:end]
    records = np.zeros((0, 4), dtype=str)
    n_type1, n_type2 = 0, 0
    # 1. 基于支线获得模糊时间戳
    stamps_over_app = get_stamps_over_load(apps, threshs[app_abb])
    print("xx")
    for event_clz in ("1", "2"):
        thresh = threshs[app_abb]
        size = sizes[house_id][app_abb][int(event_clz)-1]
        print("xxx")
        for stamp_over_app in stamps_over_app[event_clz]:
            # 2. 在总线上找到距离模糊时间戳最近时间戳
            pre_pos = np.searchsorted(mains[:, 0], stamp_over_app)
            pre_stamp = mains[pre_pos, 0]
            if abs(pre_stamp - stamp_over_app) > 60:
                # 两者相差60s以上
                continue
            # 3. 在总线上寻找最近时间戳
            search_range = np.arange(pre_pos - search_length, pre_pos + search_length)
            amps = np.array([mains[idx + size, 1] - mains[idx, 1] for idx in search_range])
            if event_clz == "1":
                candidate_poses = search_range[np.nonzero(amps > thresh)[0]]
            else:
                candidate_poses = search_range[np.nonzero(amps < -thresh)[0]]
            # 重叠的多个事件仅保留第一个
            candidate_poses = candidate_poses[np.diff(candidate_poses, prepend=float("-inf")) > size]
            if len(candidate_poses) > 0:
                n_type1 += 1
                offests = np.abs(mains[candidate_poses, 0] - stamp_over_app)
                stamp = mains[candidate_poses[np.argmin(offests)], 0]
                type_ = 1
            else:
                # 记录最近时间戳并标记为“未匹配”
                n_type2 += 1
                stamp = pre_stamp
                type_ = 2

            precision = 1 if set_name == "ukdale" else 0
            stamp = f"{stamp:.{precision}f}"
            stamp_over_app = f"{stamp_over_app:.0f}"
            record = np.array([stamp, event_clz, type_, stamp_over_app], dtype=str)
            records = np.concatenate([records, record[None, :]])
    # sort the records on stamps and save records
    records = records[np.argsort(records[:, 0])]
    save_dir = Path("PEAN") / set_name / f'house_{house_id}' # Precise Event Annotation of NILM 
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(save_dir / f"{app_abb}.csv", records, fmt="%s")
    return n_type1, n_type2

def get_stamps_over_load(apps, thresh, min_on: int = 2, min_off: int = 2):
    """ 通过阈值检测电器的开启/关闭事件的所有时间戳。 """
    # 1. 获得每个点的开关状态
    status = apps[:, 1] >= thresh

    # 2. 获得状态切换的位置（前点）
    status_diff = np.diff(status)
    event_ids = np.nonzero(status_diff)[0]
    if status[0]: event_ids = event_ids[1:]
    if status[-1]: event_ids = event_ids[:-1]
    event_ids = event_ids.reshape((-1, 2))
    on_events, off_events = event_ids[:, 0], event_ids[:, 1]

    # 3. 确保检测到的事件符合最小关闭时长和开启时长的要求
    if len(on_events) == 0: return np.array([]), np.array([])
    off_dura = on_events[1:] - off_events[:-1]
    on_events = np.concatenate([on_events[0:1], on_events[1:][off_dura > min_off]])
    off_events = np.concatenate([off_events[:-1][off_dura > min_off], off_events[-1:]])
    on_dura = off_events - on_events
    on_events = on_events[on_dura > min_on]
    off_events = off_events[on_dura > min_on]
    return {"1": apps[on_events, 0], "2": apps[off_events, 0]}

if __name__ == "__main__":
    sum_type1, sum_type2 = 0, 0
    for house_id in (1, 2, 5):
        for app_abb in "kmdwf":
            n_type1, n_type2 = annotate("ukdale", house_id, app_abb)
            sum_type1 += n_type1
            sum_type2 += n_type2
            print(n_type1, n_type2, n_type2 / (n_type1 + n_type2))
    print(sum_type1, sum_type2, sum_type2 / (sum_type1 + sum_type2))
