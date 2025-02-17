"""
MIT License
Copyright (c) 2024-present lily.

written by lily
email: lily231147@gmail.com
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import dataset


np.random.seed(42)
plt.rc("font", family="Times New Roman")

WINDOW_LENGTH = 1024
WINDOW_STRIDE = 256
N = 3
dir = Path("C:\\Users\\21975\Downloads\\nilm_lf")

def show(data_dir, set_name, house_id, app_abb, channel, n):
    """ save examples """
    print(f"Need to get {n} examples for {set_name}{house_id}-{app_abb}{channel}")
    examples = []
    apps = dataset.read(data_dir, set_name, house_id, app_abb, channel).to_numpy(dtype=np.float32)[:, 0]
    def on_click(event):
        if not event.dblclick:
            return
        center = int(event.xdata)
        left = center - WINDOW_LENGTH // 2
        right = center + WINDOW_LENGTH // 2
        ax.set_xlim(left, right)
        fig.canvas.draw()
        examples.append(apps[left:right])
        print(f"Selected {len(examples)} examples")
        if len(examples) == n:
            fig.canvas.mpl_disconnect(cid)
            plt.close(fig)
    fig = plt.figure(figsize=(5, 3), dpi=300)
    ax = plt.gca()
    plt.ylabel("Power(W)")
    plt.xlabel("Time(6s)")
    plt.plot(apps, label=app_abb)
    plt.legend(loc="upper right")
    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
    assert len(examples) == n
    return examples


if __name__ == '__main__':
    for app_abb, house_channels in dataset.ukdale_channels.items():
        for house_id, channels in house_channels.items():
            examples = []
            for i, channel in enumerate(channels):
                examples.extend(show(dir, 'ukdale', house_id, app_abb, channel, (N+i) // len(channels)))
    # for app_abb, house_channels in dataset.refit_channels.items():
    #     for house_id, channels in house_channels.items():
    #         examples = []
    #         for i, channel in enumerate(channels):
    #             examples.extend(show(dir, 'refit', house_id, app_abb, channel, (N+i) // len(channels)))


