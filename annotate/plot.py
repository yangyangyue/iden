# Copyright 2023 lily. All rights reserved.
#
# Author: lily
# Email: lily231147@proton.me


import click
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path

from utils import read_md, read_loads, read_mains

np.random.seed(42)
plt.rc("font", family="Times New Roman")

WINDOW_LENGTH = 2048
HALF_LENGTH = WINDOW_LENGTH // 2
LOADS_LENGTH = WINDOW_LENGTH // 6


def plot_curve(
    power: np.ndarray,
    title: str = "",
    xlabel: str = "Time(s)",
    idx: int = None,
    ids: list = [],
    save_path: Path = None,
    is_show: bool = True,
):
    """
    Plot curve with given power, title, xlable, idx, ids, and save figure if a save_path is provided.

    Args:
        power (`np.adarray` of shape `(_,)`): The power to be ploted.
        title (`str`): The title of figure.
        xlabel (`str`): The lable of X-axis.
        idx: (`int`): The position of target event, where a red vline will be drawn.
        ids: (`list`): The position of neighbor events, where green vlines will be drawn.
        save_path (`Path`): The path to save figure.
        is_show (`bool`): Whether to show the figure.
    """
    fig = plt.figure(figsize=(4, 2.5), dpi=300)
    plt.ylabel("Power(W)")
    plt.xlabel(xlabel)
    # plot power curve
    fig.plot(power)
    # mark target event
    target = idx and fig.axvline(idx, color="r", linestyle="--", label="target event")
    # mark neighbor events
    neighbor = None
    for id in ids:
        if id != idx:
            neighbor = fig.axvline(id, color="r", linestyle="--", label="event")
    # legend, title, save and show
    handles = [handle for handle in [neighbor, target] if handle]
    handles and fig.legend(handles=handles, loc="upper right")
    title and fig.title(title)
    save_path and fig.savefig(save_path)
    is_show and fig.show()
    plt.close(fig)


def do_plot_annotations(set_name, house, app_name, event_type):
    """
    Plot power curve around each record in annotations.

    Args:
        set_name (`str`): the name of dataset, i.e., UKDALE or REDD.
        house (`str`): the house name in the dataset, liking house_x.
        app_name (`str`): the name of appliance in the house.
        event_type (`int`): if event_type == 1 (matched event), then plot 100 matched events randomly;
                            if event_type == 2 (unmatched event), then plot all unmatched events in order.
    """
    if app_name not in read_md(Path("data"))[set_name][house].keys():
        return
    mains = read_mains(set_name, house)
    anns = np.loadtxt(Path("annotate") / set_name / house / f"{app_name}.csv")
    anns_of_type = anns[anns[:, 2] == event_type]
    if event_type == 1:
        # select 100 annotations of type1 randomly.
        anns_of_type = anns_of_type[np.random.permutation(len(anns_of_type))[:100]]
    for i, (stamp, event_clz, _, stamp_over_load) in enumerate(anns_of_type):
        print(f"{set_name}-{house}-{app_name}-{i}-{stamp}")
        # slice the window of mains
        idx = np.nonzero(mains[:, 0] == stamp)[0][0]
        begin = max(idx - HALF_LENGTH, 0)
        end = min(idx + HALF_LENGTH, len(mains))
        mains_in_window = mains[begin:end]
        # calc the idx of target annotations and the ids of neighbor annotations
        target_event_idx = np.nonzero(mains_in_window[:, 0] == stamp)[0][0]
        event_ids = np.nonzero(np.isin(mains_in_window[:, 0], anns[:, 0]))[0]
        neighbor_event_ids = event_ids[event_ids != target_event_idx]
        # plot
        plot_curve(
            mains_in_window[:, 1],
            title=f"{stamp}--{event_clz}--{mains[idx, 1]}--{stamp_over_load}",
            idx=target_event_idx,
            ids=neighbor_event_ids,
        )


@click.group()
def cli():
    ...


@cli.command()
@click.option("-t", "--event-type", default=0, help="Input what kind of event to plot")
def annotation(event_type):
    plt.figure(figsize=(4, 2.5), dpi=300)
    md = read_md()
    for set_name in ("ukdale", "redd"):
        for house in md[set_name].keys():
            for app_name in md[set_name][house].keys():
                do_plot_annotations(set_name, house, app_name, event_type=event_type)


@cli.command()
@click.option("-s", "--set-name", default="ukdale", help="Input UKDALE or REDD")
@click.option("-h", "--house", default="house_1", help="Input house of the dataset")
@click.option("-a", "--app-name", default="kettle", help="Input the appliance name")
def example(set_name, house, app_name):
    """
    Show and save event examples. First, find the event timestamps over load signal through pre-determined
    threshold, then regard the nearest timestamps over mains as event position.

    Args:
        set_name (`str`): the name of dataset, i.e., UKDALE or REDD.
        house (`str`): the house name in the dataset, liking house_x.
        app_name (`str`): the name of appliance in the house.
    """
    # load metadata, mains and load signal
    app_md = read_md(Path("data"))[set_name][house][app_name]
    mains = read_mains(set_name, house)[:, :2]
    apps = read_loads(set_name, house, app_name)
    apps = apps[np.searchsorted(apps[:, 0], mains[0, 0]) :]
    # show a sliding windiw once
    for apps_in_window in sliding_window_view(apps, (LOADS_LENGTH, 2))[::LOADS_LENGTH]:
        apps_in_window = apps_in_window.reshape((LOADS_LENGTH, 2))
        # 1. get event position of each event class from load signal
        state = apps_in_window[:, 1] > app_md["threshold"]
        event_ids = np.nonzero(np.diff(state))[0]
        if not event_ids:
            # no event exist in this window
            continue
        # 2. get the sliding window over mains and match the nearest position of events over mains
        mid = np.searchsorted(mains[:, 0], apps_in_window[0, len(apps_in_window) // 2])
        mains_in_window = mains[mid - HALF_LENGTH : mid + HALF_LENGTH]
        event_ids = np.searchsorted(mains_in_window[:, 0], apps_in_window[event_ids, 0])
        # 3. show and save
        dir = Path("examples") / set_name / house
        dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(dir / f"{app_name}.csv", mains_in_window, fmt="%.2f")
        np.savetxt(dir / f"{app_name}_app.csv", apps_in_window, fmt="%.2f")
        save_path = dir / f"{app_name}_label.png"
        plot_curve(mains_in_window[:, 1], ids=event_ids, save_path=save_path)


@cli.command()
@click.option("-s", "--set-name", default="ukdale", help="Input UKDALE or REDD")
@click.option("-h", "--house", default="house_1", help="Input house of the dataset")
@click.option("-a", "--app-name", default="kettle", help="Input the appliance name")
@click.option("-t", "--thresh", default=60, help="Input the threshold")
def app(set_name, house, app_name, thresh):
    """plot load signal using sliding window, plot threshold as well"""
    apps = read_loads(set_name, house, app_name)
    for apps_in_window in sliding_window_view(apps, (LOADS_LENGTH, 2))[::LOADS_LENGTH]:
        apps_in_window = apps_in_window.reshape((LOADS_LENGTH, 2))
        state = apps_in_window[:, 1] > thresh
        event_ids = np.nonzero(np.diff(state))[0]
        if len(event_ids) > 0:
            print(apps_in_window[event_ids, 0])
            plot_curve(apps_in_window[:, 1], xlabel="Time(6s)", ids=event_ids)


if __name__ == "__main__":
    cli()
