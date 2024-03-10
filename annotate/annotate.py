# Copyright 2023 lily. All rights reserved.
#
# Author: lily
# Email: lily231147@proton.me


from pathlib import Path
import shutil

import numpy as np

from utils import read_md, read_loads, read_mains


def get_stamps_over_load(
    apps: np.ndarray, app_md: dict, min_on: int = 2, min_off: int = 2
) -> dict:
    """
    Detect all timestamps for ON/OFF event of appliance through threshold.
    Ensure that the detected events comply with the min off-duration and on-duration requirements,
    which is intended to avoid false detections caused by anomalous singularities.

    Args:
        apps (`np.ndarray` of shape `(_, 2)`): The sequence of timestamps and powers.
        app_md (`dict`): The metadata of appliance, i.e., the pre-determined threshold.
        min_on (`int`): The minimum on-duration (6s).
        min_off (`int`): The minimum off-duration (6s).
    Returns:
        dict: the timestamps for ON/OFF event.
    """
    # 1. get the on/off status of each point
    status = apps[:, 1] >= app_md["threshold"]

    # 2. get the on/off event pre-index
    status_diff = np.diff(status)
    event_ids = np.nonzero(status_diff)[0]
    if status[0]:
        event_ids = event_ids[1:]
    if status[-1]:
        event_ids = event_ids[:-1]
    event_ids = event_ids.reshape((-1, 2))
    on_events, off_events = event_ids[:, 0], event_ids[:, 1]

    # 3. filter the on/off events to guarantee all on/off duration greater than its min duration thresh
    if len(on_events) == 0:
        return np.array([]), np.array([])
    off_dura = on_events[1:] - off_events[:-1]
    on_events = np.concatenate([on_events[0:1], on_events[1:][off_dura > min_off]])
    off_events = np.concatenate([off_events[:-1][off_dura > min_off], off_events[-1:]])
    on_dura = off_events - on_events
    on_events = on_events[on_dura > min_on]
    off_events = off_events[on_dura > min_on]
    return {"1": apps[on_events, 0], "2": apps[off_events, 0]}


def annotate(
    set_name: str, house: str, app_name: str, search_length: int = 30
) -> tuple:
    """
    Annotate the event timestamps over mains (sampled at 1s) of UKDALE and REDD. In short, we first find the event
    timestamps over load signal through pre-determined threshold, then match the specific event on a scope over mains.

    For each event, we annotate 4 items, i.e. timestamp over mains, event class, event type and timestamp over load signal.
    Event class is either 1 (on event) and 2 (off event). Event type is either 1 (matched event) or 2 (unmatched event).

    Note that there are 3 possible scenarios of faulty annotation:
        1. Events don't exist on load signal, but do exist on mains, which can lead to missed labels.
        2. Events exist on load signal, but are not found on mains, which can be attributed to the events overlapping,
           large offset between load signal and mains, or acquisition issues. The type of these events is assigned as 1.
        3. Matching the wrong events on the mains.
    Taking into account the small percentage of the above scenarios, we just ignore these issues!


    Args:
        set_name (`str`): the name of dataset, i.e., UKDALE or REDD.
        house (`str`): the house name in the dataset, liking house_x.
        app_name (`str`): the name of appliance in the house.
        search_length (`int`): The search radius centered at the timestamp over load signal.
    Returns:
        tuple: The number of type1 events and type2 events.
    """
    # load data and slice load signal in the range of mains
    app_md = read_md()[set_name][house].get(app_name)
    if not app_md:
        return 0, 0
    mains = read_mains(set_name, house)
    apps = read_loads(set_name, house, app_name)
    begin = np.searchsorted(apps[:, 0], mains[0, 0])
    end = np.searchsorted(apps[:, 0], mains[-1, 0])
    apps = apps[begin:end]
    records = np.zeros((0, 4), dtype=str)
    n_type1, n_type2 = 0, 0
    # 1. locate event stamps from load signal
    stamps_over_app = get_stamps_over_load(apps, app_md)
    for event_clz in ("1", "2"):
        thresh = app_md["threshold"]
        size = app_md["sizes"][event_clz]
        for stamp_over_app in stamps_over_app[event_clz]:
            # 2. locate the nearest pos and its timestamp over mains to the timestamp over load signal
            pre_pos = np.searchsorted(mains[:, 0], stamp_over_app)
            pre_stamp = mains[pre_pos, 0]
            if abs(pre_stamp - stamp_over_app) > 60:
                # ignore timestamp in the interval
                continue
            # 3. match the timestamp of nearest target event within a scope over mains
            search_range = np.arange(pre_pos - search_length, pre_pos + search_length)
            amps = np.array(
                [
                    mains[idx + (size + 1) // 2, 1] - mains[idx - size // 2, 1]
                    for idx in search_range
                ]
            )
            if event_clz == "1":
                candidate_poses = search_range[np.nonzero(amps > thresh)[0]]
            else:
                candidate_poses = search_range[np.nonzero(amps < -thresh)[0]]
            # remove overlop events, only restore the last one
            candidate_poses = candidate_poses[
                np.diff(candidate_poses, append=float("inf")) > size
            ]
            if len(candidate_poses) > 0:
                n_type1 += 1
                offests = np.abs(mains[candidate_poses, 0] - stamp_over_app)
                stamp = mains[candidate_poses[np.argmin(offests)], 0]
                type_ = 1

            else:
                # record the nearest stamp over mains and mark the event as unmatched if not find a target event
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
    save_dir = Path("EAMN") / set_name / house / "annotation"
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(save_dir / f"{app_name}.csv", records, fmt="%s")
    return n_type1, n_type2


def copy_mains(set_name: str, house: str):
    """
    move `mains.dat` of UKDALE or `channel_1.dat`, `channel_2.dat` of REDD to `EAMN`.

    Args:
        set_name (`str`): the name of dataset, i.e., UKDALE or REDD.
        house (`str`): the house name in the dataset, liking house_x.
    """
    source_dir = Path("data") / set_name / house
    target_dir = Path("EAMN") / set_name / house
    if set_name == "ukdale":
        source_path = source_dir / "mains.dat"
        target_path = target_dir / "mains.dat"
        if not target_path.exists():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source_path, target_path)
    else:
        source_path = source_dir / "channel_1.dat"
        target_path = target_dir / "channel_1.dat"
        if not target_path.exists():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source_path, target_path)
        source_path = source_dir / "channel_2.dat"
        target_path = target_dir / "channel_2.dat"
        if not target_path.exists():
            shutil.copy(source_path, target_path)


if __name__ == "__main__":
    md = read_md()
    sum_type1, sum_type2 = 0, 0
    for set_name in ("ukdale", "redd"):
        for house in md[set_name].keys():
            copy_mains(set_name, house)
            for app_name in md[set_name][house].keys():
                n_type1, n_type2 = annotate("ukdale", house, app_name)
                sum_type1 += n_type1
                sum_type2 += n_type2
                print(n_type1, n_type2, n_type2 / (n_type1 + n_type2))
    print(sum_type1, sum_type2, sum_type2 / (sum_type1 + sum_type2))
