"""
直接画出结果
"""

import sys
sys.path.append('..')
from dataset import read, sizes

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
plt.rc("font", family="Times New Roman")

house_id=2

mains = read("ukdale", house_id)
apps = read("ukdale", house_id, 'k')

for app_abb in 'kmdwf':
    valid = np.loadtxt(Path("..") / 'PEAN' / 'ukdale' / f'house_{house_id}' / f"{app_abb}-valid.csv")
    start_idx = np.searchsorted(valid[:, 0], mains[:, 0], side='right') - 1
    end_idx = np.searchsorted(valid[:, 1], mains[:, 0], side='right')
    mains = mains[np.where(start_idx == end_idx)[0]]
    # path = Path("..") / "PEAN" / "ukdale" / f"house_{house_id}" / f"{app_abb}.csv"
    pred_anns = np.loadtxt(f'sl-ukdale15-ukdale2-{app_abb}.csv')
    gt_anns = np.loadtxt(Path("..") / 'PEAN' / 'ukdale' / f'house_{house_id}' / f"{app_abb}.csv")
    gt_anns = gt_anns[gt_anns[:, 2] == 1]
    predf = np.loadtxt(f'sl-ukdale15-ukdale2-{app_abb}.csv-predf')
    gtf = np.loadtxt(f'sl-ukdale15-ukdale2-{app_abb}.csv-gtf')
    pred_ids = [np.nonzero(mains[:,0] == item[0])[0][0] for item in pred_anns]
    plt.plot(mains[:, 1])
    # plt.plot(apps[:, 0], apps[:, 1])
    for i, idx in enumerate(pred_ids):
        size = sizes[house_id][app_abb][0 if pred_anns[i, 1] == 1 else 1]
        if mains[idx, 0] in predf:
            plt.axvline(idx-size//2, color='g', linestyle='--')
            plt.axvline(idx+(size+1)//2, color='g', linestyle='--')
        else:
            plt.axvline(idx-size//2, color='g')
            plt.axvline(idx+(size+1)//2, color='g')
    for stamp in gt_anns[:, 0]:
        idx= np.where(mains[:, 0] == stamp)[0][0]
        if stamp in gtf:
            plt.axvline(idx-size//2, color='r', linestyle='--')
            plt.axvline(idx+(size+1)//2, color='r', linestyle='--')
        else:
            plt.axvline(idx-size//2, color='r')
            plt.axvline(idx+(size+1)//2, color='r')
    plt.show()