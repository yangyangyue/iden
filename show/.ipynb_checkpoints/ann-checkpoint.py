"""
MIT License
Copyright (c) 2024-present lily.

written by lily
email: lily231147@gmail.com
"""
import sys
sys.path.append('..')
from dataset import read, sizes

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


np.random.seed(42)
plt.rc("font", family="Times New Roman")

# def plot(house, mains, ann, app, sizes):
#     event = ann[np.random.choice(range(len(ann)))]
#     idx= np.where(mains[:, 0] == event[0])[0][0]
#     size = sizes[0 if event[1] == 1 else 1]
#     mains = mains[idx-15:idx+15, 1]
#     fig = plt.figure(figsize=(5, 3), dpi=300)
#     plt.plot(mains, label='mains')
#     plt.axvline(15-size//2, color='r', label=app)
#     plt.axvline(15+(size+1)//2, color='r')
#     plt.ylabel("Power(W)")
#     plt.xlabel("Time(s)")
#     plt.legend(loc="upper right")
#     # plt.show()
#     plt.savefig(f'figs/{house}-{app}.png', bbox_inches='tight', dpi=300)
#     plt.close(fig)


# if __name__ == "__main__":
#     for house_id in (1, 2, 5):
#         mains = read("ukdale", house_id)
#         for app_abb in "kmdwf":
#             path = Path("..") / "PEAN" / "ukdale" / f"house_{house_id}" / f"{app_abb}.csv"
#             ann = np.loadtxt(path)
#             plot(house_id, mains, ann, app_abb, sizes[house_id][app_abb])

plt.rcParams['font.size'] = 16


x = [read("ukdale", 1), read("ukdale", 2), read("ukdale", 5)]

fig, axs = plt.subplots(10, 3, figsize=(12, 19.8))
i,j =0,0
for app_abb in 'kmdwf':
    for is_on in (1, 2):
        for house_id in (1, 2, 5):
            mains = x[house_id//2]
            size = sizes[house_id][app_abb][0 if is_on == 1 else 1]
            for k in range(1):
                path = Path("..") / "PEAN" / "ukdale" / f"house_{house_id}" / f"{app_abb}.csv"
                ann = np.loadtxt(path)
                ann = ann[ann[:, 1] == is_on]
                event = ann[np.random.choice(range(len(ann)))]
                idx= np.where(mains[:, 0] == event[0])[0][0]
                w_mains = mains[idx-15:idx+15, 1]
                # axs[i, j].set_ylim(0, 4000)
                if i == 0 and j == 2:
                    axs[i, j].plot(w_mains, label='mains')
                    axs[i, j].axvline(15-size//2, color='r', label="event")
                    axs[i, j].axvline(15+(size+1)//2, color='r')
                else:
                    axs[i, j].plot(w_mains)
                    axs[i, j].axvline(15-size//2, color='r')
                    axs[i, j].axvline(15+(size+1)//2, color='r')
                if i!=9: axs[i, j].set_xticks([])
                if i == 9 and j==1:axs[i, j].set_xlabel("Time(s)")
                # if j==0:axs[i, j].set_ylabel("Power(W)")
                j = (j+1)%3
        i += 1
# for i, app in enumerate(['KT', 'MW', 'DW', 'WM', 'FD']):
axs[0, 2].legend()
# fig.legend(handles=[l1, l2, l3], loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.05))
for j, house in enumerate(['HOUSE_1', 'HOUSE_2', 'HOUSE_5']):
    axs[0, j].set_title(f"{house.upper()}", fontsize=16, loc='center', y=1.02)
    # axs[0, 2*j+1].set_title(f"{house.upper()}-2", fontsize=16, loc='center', y=1.02)
for i, app in enumerate(['KT', 'MW', 'DW', 'WM', 'FD']):
    axs[2*i, 0].set_ylabel(f"{app}-ON", fontsize=16, rotation=90, labelpad=12)
    axs[2*i+1, 0].set_ylabel(f"{app}-OFF", fontsize=16, rotation=90, labelpad=12)

fig.text(-0.08, 0.99, 'Power(W)')

plt.subplots_adjust(wspace=0.15, hspace=0.1, left=0, right=1, bottom=0, top=1)
# plt.show()


plt.savefig(f'figs/all_ann.png', dpi=600, bbox_inches='tight')
plt.close()

