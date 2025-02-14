"""
MIT License
Copyright (c) 2024-present lily.

written by lily
email: lily231147@gmail.com
"""

import matplotlib.pyplot as plt
import numpy as np

from read_data import read_md, read_annotation, read_mains

np.random.seed(42)
plt.rc("font", family="Times New Roman")

def show(house, mains, ann, app, sizes):
    event = ann[np.random.choice(range(len(ann)))]
    idx= np.where(mains[:, 0] == event[0])[0][0]
    size = sizes["1" if event[1] == 1 else "2"]
    mains = mains[idx-15:idx+15, 1]
    fig = plt.figure(figsize=(5, 3), dpi=300)
    ax = plt.gca()
    plt.plot(mains, label='mains')
    plt.axvline(15-(size+1)//2, color='r', label=app)
    plt.axvline(15+(size)//2, color='r')
    plt.ylabel("Power(W)")
    plt.xlabel("Time(s)")
    plt.legend(loc="upper right")
    # plt.show()
    plt.savefig(f'{house}-{app}.png', bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    md = read_md()
    for house in md['ukdale'].keys():
        mains = read_mains('ukdale', house)
        for app_name in md['ukdale'][house].keys():
            sizes = md['ukdale'][house][app_name]['sizes']
            ann = read_annotation('ukdale', house, app_name)
            show(house, mains, ann, app_name, sizes)


