"""
MIT License
Copyright (c) 2024-present lily.

written by lily
email: lily231147@gmail.com
"""

import hashlib
from pathlib import Path
import random
import re
import tempfile

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

"""
本试验前提条件：假定同一个电器的同类事件持续时间相同，而且是已知的
那么，实际上只需要输出一个事件位置 而无需关心事件长度
如果使用事件中点代表事件位置的话，由于事件的波动趋势并不相同，所以这个中点没有代表
也就是在实际场景中，有时很难说事件过程中哪个点更中
特殊地，如果输出的两个事件重叠了，也无法说明哪个更好
所以，统一起见，以事件开始的点代替事件位置，事件重叠时取最后一个
"""

random.seed(42)

# 全局变量
# DIR = Path('/root/autodl-tmp/nilm_lf')
DIR = Path('C://Users//21975//Downloads//nilm_lf')
ids = {"k": 0, "m": 1, "d": 2, "w": 3, "f": 4}
# threshs ={"k": 2000, "m": 200, "d": 10, "w": 20, "f": 50} 主要是w的阈值差异 1200是我个人设定的
threshs = {"k": 2000, "m": 1200, "d": 1200, "w": 1200, "f": 50}
ceils = {"k": 3100, "m": 3000, "d": 2500, "w": 2500, "f": 300}
sizes = {
    1: {'k': (2, 2), 'm': (5, 2), 'd': (2, 2), 'w': (3, 3), 'f': (3, 2)},
    2: {'k': (2, 2), 'm': (4, 2), 'd': (2, 2), 'w': (2, 2), 'f': (4, 2)},
    5: {'k': (2, 2), 'm': (6, 2), 'd': (2, 2), 'w': (2, 2), 'f': (4, 2)}
}
names = ['kettle', 'microwave', 'dishwasher', 'washing_machine', 'fridge']

WINDOW_SIZE = 1024
WINDOW_STRIDE = 512

# 不考虑refit，因为其采样率是1/8s
ukdale_channels = {
    'k': {1: [10], 2: [8], 5: [18]},
    'm': {1: [13], 2: [15], 5: [23]},
    'd': {1: [6], 2: [13], 5: [22]},
    'w': {1: [5], 2: [12], 5: [24]},
    'f': {1: [12], 2: [14], 5: [19]}
}

def read(set_name, house_id, app_abb=None, channel=None):
    """ 
    读取总线(1s)或者支线(6s) 
    powers: [[f64], [f64]]
    """
    assert set_name ==  'ukdale', 'currently only support ukdale dataset'
    if not app_abb: 
        path = DIR / set_name / f'house_{house_id}' / f'mains.dat'
    else: 
        if channel is None: channel = ukdale_channels[app_abb][house_id][0]
        path = DIR / set_name / f'house_{house_id}' / f'channel_{channel}.dat'

    temp_dir = Path(tempfile.gettempdir())
    temp_path = temp_dir / hashlib.sha256(str(path).encode()).hexdigest()
    if temp_path.exists():
        powers = np.load(temp_path)
    else:
        df = pd.read_csv(path, sep=" ", header=None).iloc[:, :2]
        powers = df.to_numpy()
        powers[powers[:, 1]<5, 1] = 0
        upper=6000 if not app_abb else ceils[app_abb]
        powers[powers[:, 1]>upper, 1] = upper
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(temp_path, powers)
    return powers

def load_data(set_name, house_ids, app_abb, stage):
    """ 加载指定房屋的数据，包括总线、时间戳、事件位置和类别 """
    stamps_list, powers_list, poses_list, clzes_list = [], [], [], []
    for house_id in house_ids:
        mains = read(set_name, house_id)
        anns = np.loadtxt(Path("PEAN") / set_name / f'house_{house_id}' / f"{app_abb}.csv")
        # ukdale 1 在训练时只取前0.15的数据 否则太耗时了
        if stage == 'fit' and set_name == 'ukdale' and house_id == 1: mains = mains[0: int(0.15 * len(mains))]
        # 只考虑匹配事件
        anns = anns[anns[:, 2] == 1]
        for win in np.lib.stride_tricks.sliding_window_view(mains, (WINDOW_SIZE, 2))[:, 0][::WINDOW_STRIDE]:
            # 获得每个窗口对应的标签
            win_anns = anns[(anns[:, 0] >= win[0, 0]) & (anns[:, 0] <= win[-1, 0])]
            stamps_list.append(stamps:=win[:, 0])
            powers_list.append(win[:, 1].astype(np.float32))
            poses_list.append(np.nonzero(np.isin(stamps, win_anns[:, 0]))[0])
            clzes_list.append(win_anns[:, 1].astype(np.int64))
    return stamps_list, powers_list, poses_list, clzes_list

class ApplianceDataset(Dataset):
    """ 数据集 对应一类设备 """
    def __init__(self, app_abb, stamps_list, powers_list, poses_list, clzes_list, stage):
        super().__init__()
        self.app_abb = app_abb
        self.stamps_list = stamps_list # [[np.float64]]
        self.powers_list = powers_list # [[np.float32]]
        self.poses_list = poses_list # [[np.int64]]
        self.clzes_list = clzes_list # [[np.int64]]
        if stage == 'fit': self.balance()

    def __getitem__(self, index):
        """ 返回一个样本 其中stamps和powers是长度为L的序列 poses和clzes是长度为T的序列 """
        return ids[self.app_abb], self.stamps_list[index], self.powers_list[index], self.poses_list[index], self.clzes_list[index]

    def __len__(self):
        return len(self.stamps_list)

    def balance(self, pos_neg_ratio=1/3):
        """ 正负样本平衡 不限制正样本数量是因为有fridge这种全正电器 """
        samples = np.array([len(clzes) > 0 for clzes in self.clzes_list])
        n_pos = np.sum(samples)
        n_neg = len(samples) - n_pos
        if n_pos / pos_neg_ratio >= n_neg: return
        n_neg = int(n_pos / pos_neg_ratio)
        pos_ids = np.nonzero(samples)[0]
        neg_ids = np.nonzero(samples == False)[0]
        neg_ids = neg_ids[np.random.permutation(len(neg_ids))[:n_neg]]
        keep_ids = np.sort(np.concatenate([pos_ids, neg_ids]))
        self.stamps_list = [self.stamps_list[idx] for idx in keep_ids]
        self.powers_list = [self.powers_list[idx] for idx in keep_ids]
        self.poses_list = [self.poses_list[idx] for idx in keep_ids]
        self.clzes_list = [self.clzes_list[idx] for idx in keep_ids]

def get_sets(set_houses, stage):
    """ 获取多个房屋中各电器的数据集 """
    datasets = [[], [], [], [], []]
    match = re.match(r'^(\D+)(\d+)$', set_houses)
    set_name, house_ids = match.groups()
    return [ApplianceDataset(app_abb, *load_data(set_name, house_ids, app_abb, stage), stage) for app_abb in "kmdwf"]
