# Copyright 2023 lily. All rights reserved.
#
# Author: lily
# Email: lily231147@proton.me


import hashlib
import json
import tempfile
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch import FloatTensor, LongTensor
from torch.utils.data import DataLoader, Dataset


WINDOW_LENGTH = 2048
WINDOW_STRIDE = 512


def read_md() -> dict:
    """Read metadate drom `EAMN` floder."""
    with open(Path("EAMN") / "metadata.json", "r") as file:
        md = json.load(file)
    return md


def merge(signal1: pd.DataFrame, signal2: pd.DataFrame) -> pd.DataFrame:
    """Merge 2 Dataframes on timestamp, and then add their power."""
    signal = pd.merge(signal1, signal2, on="stamp")
    signal.iloc[:, 1] = signal.iloc[:, 1] + signal.iloc[:, 2]
    return signal.iloc[:, :2]


def read_df(path: Path) -> pd.DataFrame:
    """If cached, read dataframe from cache, or read dataframe from path and cache it otherwise"""
    temp_dir = Path(tempfile.gettempdir())
    father_path = temp_dir / hashlib.sha256(str(path).encode()).hexdigest()
    if father_path.exists():
        df = pd.read_feather(father_path)
    else:
        father_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(path, sep=" ", header=None).iloc[:, :2]
        df.columns = ["stamp", "power"]
        df.sort_values(by="stamp")
        df.to_feather(father_path)
    return df


def read_mains(set_name: str, house: str) -> np.ndarray:
    """Read mains for house from dataset. For REDD, add `channel_1` and `channel_2`"""
    dir = Path("EAMN") / set_name / house
    if set_name == "redd":
        # read and merge mains of redd
        mains_1 = read_df(dir / "channel_1.dat")
        mains_2 = read_df(dir / "channel_2.dat")
        signal = merge(mains_1, mains_2)
    else:
        # read mains of ukdale
        signal = read_df(dir / "mains.dat")
    return signal.values


def read_loads(set_name: str, house: str, app_name: str) -> np.ndarray:
    """Read loads signal. It is required to place raw data in `data` floder"""
    dir = Path("data") / set_name / house
    md = read_md()
    channel = md[set_name][house][app_name]["channel"]
    if isinstance(channel, list):
        signal = read_df(dir / f"channel_{channel[0]}.dat")
        for sub_channel in channel[1:]:
            new_signal = read_df(dir / f"channel_{sub_channel}.dat")
            signal = merge(signal, new_signal)
    else:
        signal = read_df(dir / f"channel_{channel}.dat")
    return signal.values


def read_annotation(set_name: str, house: str, app_name: str) -> np.ndarray:
    """
    Read annotation for specific appliance.

    Returns:
        (`_, 4`): annotation whose columns are timestamps over mains, event classes,
        event types and timestamps over load signal.
    """
    path = Path("EAMN") / set_name / house / "annotation" / f"{app_name}.csv"
    return np.loadtxt(path)


def read_data(
    set_name: str, houses: list[str], house_ratios: list[float], app_name: str
) -> tuple[
    list[np.ndarray[np.float64]],
    list[np.ndarray[np.float32]],
    list[np.ndarray[np.int32]],
    list[np.ndarray[np.int32]],
]:
    """
    Load data to build dataset. The original data is slided into sliding windows based on the `length` and `stride`.

    Args:
        set_name: the name of dataset, i.e., UKDALE or REDD.
        houses: the name of the house adopted
        house_ratios: the proportion of data used in each house.
        app_name: the name of appliance in the houses.

    Returns:
        stamps_list: a list of timestamps of each point in the window
        powers_list: a list of powers of each point in the window
        poses_list: a list of positions of each event in the window
        clzes_list: a list of classes of each event in the window
    """
    stamps_list = []
    powers_list = []
    poses_list = []
    clzes_list = []
    md = read_md()
    for house, house_ratio in zip(houses, house_ratios):
        if app_name not in md[set_name][house].keys():
            # no appliance exist in this house
            continue
        # read raw data
        mains = read_mains(set_name, house)
        anns = read_annotation(set_name, house, app_name)
        # only keep events of type1
        anns = anns[anns[:, 2] == 1]
        # perform sliding window
        windows = sliding_window_view(mains, (WINDOW_LENGTH, 2))[::WINDOW_STRIDE]
        # random select a specific number of samples
        n_windows = len(windows)
        selected = np.random.permutation(n_windows)[: int(n_windows * house_ratio)]
        for win_mains in windows[selected]:
            # get the annotations within current sliding window
            win_anns = anns[
                (anns[:, 0] >= win_mains[0, 0, 0]) & (anns[:, 0] <= win_mains[0, -1, 0])
            ]
            stamps = win_mains[0, :, 0]
            stamps_list.append(stamps)
            powers_list.append(win_mains[0, :, 1])
            poses_list.append(np.nonzero(np.isin(stamps, win_anns[:, 0]))[0])
            clzes_list.append(win_anns[:, 1])
    return stamps_list, powers_list, poses_list, clzes_list


def find_split_point(samples: np.ndarray[bool], train_ratio: float = 0.8):
    """
    Fins the split point for splitting train data and val data.
    Note that the split logic only rely on the samples with event.
    """
    # Calculate the cumulative sum of Trues
    cumsum = np.cumsum(samples)
    # Find the split point
    train_event_num = int(cumsum[-1] * train_ratio)
    split_point = np.argmax(cumsum >= train_event_num)
    return split_point


class NILMDataset(Dataset):
    r"""
    The dataset used to train or test model.

    Args:
        stamps_list: a list of timestamps of each point in the window.
        powers_list: a list of powers of each point in the window.
        poses_list: a list of positions of each event in the window.
        clzes_list: a list of classes of each event in the window.
    """

    def __init__(
        self,
        stamps_list: list[np.ndarray[np.float64]],
        powers_list: list[np.ndarray[np.float32]],
        poses_list: list[np.ndarray[np.int32]],
        clzes_list: list[np.ndarray[np.int32]],
    ):
        super(NILMDataset, self).__init__()
        self.stamps_list = stamps_list
        self.powers_list = powers_list
        self.poses_list = poses_list
        self.clzes_list = clzes_list

    def __getitem__(
        self, index
    ) -> tuple[np.ndarray, FloatTensor, LongTensor, LongTensor]:
        """
        Return a sample.

        Returns:
            stamps: L, timestamps of the item.
            powers: L, powers of the item.
            poses: T, position num of events within the sliding window.
            clzes: T, class num of events within the sliding window.
        """
        stamps = self.stamps_list[index]
        powers = torch.as_tensor(self.powers_list[index], dtype=torch.float32)
        poses = torch.as_tensor(self.poses_list[index], dtype=torch.int64)
        clzes = torch.as_tensor(self.clzes_list[index], dtype=torch.int64)
        return stamps, powers, poses, clzes

    def __len__(self):
        return len(self.stamps_list)

    def balance_samples(self, pos_neg_ratio: float = 1 / 3):
        samples = np.array([len(clzes) > 0 for clzes in self.clzes_list])
        n_pos = np.sum(samples)
        n_neg = len(samples) - n_pos
        if n_pos / n_neg < pos_neg_ratio:
            n_neg = min(n_neg, int(n_pos / pos_neg_ratio))
        else:
            n_pos = min(n_pos, int(n_neg * pos_neg_ratio))
        pos_ids = np.nonzero(samples)[0]
        pos_ids = pos_ids[np.random.permutation(len(pos_ids))[:n_pos]]
        neg_ids = np.nonzero(samples == False)[0]
        neg_ids = neg_ids[np.random.permutation(len(neg_ids))[:n_neg]]
        keep_ids = np.sort(np.concatenate([pos_ids, neg_ids]))
        self.stamps_list = [self.stamps_list[idx] for idx in keep_ids]
        self.powers_list = [self.powers_list[idx] for idx in keep_ids]
        self.poses_list = [self.poses_list[idx] for idx in keep_ids]
        self.clzes_list = [self.clzes_list[idx] for idx in keep_ids]


def collate_fn(batch):
    """organize data for a batch manually"""
    stamps_batch, powers_batch, poses_batch, clzes_batch = tuple(zip(*batch))
    powers_batch = torch.stack(powers_batch)
    return (stamps_batch, powers_batch, poses_batch, clzes_batch)


def get_loader(
    set_name: str,
    houses: list,
    house_ratios: list,
    app_name: str,
    bs: int,
    is_train: bool = True,
) -> Union[tuple[DataLoader, DataLoader], DataLoader]:
    """
    Return train loader and val loader if is_train, or return test loader otherwise.

    Args:
        set_name: the name of dataset, i.e., UKDALE or REDD.
        houses: the name of the house adopted
        house_ratios: the proportion of data used in each house.
        app_name: the name of appliance in the houses.
        bs: batch size.
        is_train: whether to return the train loader and val loader, or return the test loader

    Returns:
        train loader and val loader, or test loader
    """
    stamps_list, powers_list, poses_list, clzes_list = read_data(
        set_name, houses, house_ratios, app_name
    )
    if is_train:
        # return train loader and val loader
        samples = np.array([len(clzes) > 0 for clzes in clzes_list])
        split_point = find_split_point(samples)
        train_set = NILMDataset(
            stamps_list[:split_point],
            powers_list[:split_point],
            poses_list[:split_point],
            clzes_list[:split_point],
        )
        train_set.balance_samples()
        val_set = NILMDataset(
            stamps_list[split_point:],
            powers_list[split_point:],
            poses_list[split_point:],
            clzes_list[split_point:],
        )
        train_loader = DataLoader(
            train_set, batch_size=bs, collate_fn=collate_fn, shuffle=True
        )
        val_loader = DataLoader(val_set, batch_size=bs, collate_fn=collate_fn)
        return train_loader, val_loader
    else:
        # return test loader
        test_set = NILMDataset(stamps_list, powers_list, poses_list, clzes_list)
        test_loader = DataLoader(test_set, batch_size=bs, collate_fn=collate_fn)
        return test_loader
