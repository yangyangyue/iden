import json
import sys

sys.path.append("/home/aistudio/external-libraries")
from read_data import NILMDataset, get_datasets

from gen_net import GenConfig, Gmime
import numpy as np
from pathlib import Path

import click
from rich.console import Console
from rich.progress import (
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    Progress,
)
from rich.table import Table

import torch
from torch.utils.data import DataLoader
from utils import cal_metrics, Metric, filter_pred

BATCH_SIZE = 64


def test_one_appliance(set_name, app_name, method, info, config):
    app_idx = info["app_ids"][set_name][app_name]
    n_class = info["n_class"][set_name][app_name]
    amplitude_threshold = info["amplitude_threshold"][set_name][app_name]
    stable_threshold = info["stable_threshold"][set_name][app_name]
    # 设备的数据主目录，eg: ./data/ukdale/data8
    set_dir = Path(f"nilm_events_simple/{set_name}")
    intervals = np.loadtxt(set_dir / f"channel_{str(app_idx)}" / "interval.txt")
    print("当前设备是：", app_name, "设备号是：", app_idx, flush=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 样本形状：[N, (1,1,1024)]
    # 标签形状：[N, {idx: int, boxes: (n, 2), labels: (n,)}]
    # ps: N为batch大小, n为样本内事件数量, idx为样本索引, boxes存放样本内每个事件的起止位置, labels存放样本内每个事件的类别
    # train_loader, val_loader, test_loader = get_loaders(app_idx, set_name, config.batch_size, partition)
    _, _, test_set = get_datasets(set_dir, set_name, app_idx)
    test_loader = DataLoader(
        test_set, batch_size=config.batch_size, collate_fn=NILMDataset.collate_fn
    )
    if method == "sl":
        model = SlNet(
            config.in_channels,
            config.out_channels,
            config.length,
            n_class,
            config.label_method,
            config.backbone,
        ).to(device)
        checkpoint_dir = (
            Path("weights") / method / config.label_method / config.backbone
        )
        case_dir = (
            Path("case")
            / method
            / config.label_method
            / config.backbone
            / f"{set_name}_{app_name}"
        )
    elif method == "yolo":
        model = YoloNet(
            config.in_channels,
            config.out_channels,
            config.length,
            num_classes=n_class,
            backbone=config.backbone,
        ).to(device)
        checkpoint_dir = Path("weights") / method / config.backbone
        case_dir = Path("case") / method / config.backbone / f"{set_name}_{app_name}"
    elif method == "gen":
        model = Gmime(400, n_class).to(device)
        checkpoint_dir = Path("./weights") / method
        case_dir = Path()
    else:
        raise ValueError(
            f"{method} must in models `seq_label`, `yolo` or `transformer`"
        )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{set_name}_{app_name}.pth"
    case_dir.mkdir(parents=True, exist_ok=True)

    # 读取之前保存的权重文件
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    with torch.no_grad():
        test_metric = Metric()
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>.2f}%"),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
        ) as progress:
            loader_task = progress.add_task(
                f"{set_name}/{app_name}", total=len(test_loader)
            )
            for images, targets, stamps in test_loader:
                images = images.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                pred = model(images, targets)
                progress.update(loader_task, advance=1)
                pred = filter_pred(
                    app_idx,
                    pred,
                    images[:, 0, :],
                    stamps,
                    intervals,
                    amplitude_threshold,
                    stable_threshold,
                )
                tp, fp, fn = cal_metrics(pred, targets, images[:, 0, :], case_dir)
                test_metric.add(tp, fp, fn)
        pre, rec, f1 = test_metric.get_index()
    return (
        f"{set_name}/{app_name}",
        test_metric.tp,
        test_metric.fp,
        test_metric.fn,
        pre,
        rec,
        f1,
    )


def set_app_choices(ctx, param, value):
    if value == "ukdale":
        ctx.command.params[1].type = click.Choice(
            ["kettle", "rice-cooker", "microwave", "all"]
        )
    elif value == "redd":
        ctx.command.params[1].type = click.Choice(
            ["furnace", "washer-dryer", "microwave", "all"]
        )
    else:
        ctx.command.params[1].type = click.Choice(["all"])
    return value


@click.command()
@click.option(
    "--dataset",
    type=click.Choice(["ukdale", "redd", "all"]),
    prompt="DataSet Name",
    callback=set_app_choices,
)
@click.option(
    "--app",
    type=click.Choice(
        ["kettle", "rice-cooker", "microwave", "furnace", "washer-dryer", "all"]
    ),
    prompt="Appliance Name",
)
@click.option(
    "--method", type=click.Choice(["sl", "yolo", "transformer"]), default="sl"
)
def test(dataset, app, method):
    table = Table(title=f"the identification experiments of {method}")
    table.add_column("appliance", style="blue")
    table.add_column("tp", style="medium_purple1")
    table.add_column("fp", style="medium_purple1")
    table.add_column("fn", style="medium_purple1")
    table.add_column("pre", style="magenta")
    table.add_column("rec", style="cyan")
    table.add_column("f1", style="green")

    with open("nilm_events_simple/metadata.json", "r") as file:
        info = json.load(file)
    if method == "sl":
        config = SlConfig(batch_size=2048)
    elif method == "yolo":
        config = YoloConfig()
    else:
        config = GenConfig()
    if dataset == "all":
        # test all appliances in all datasets
        for app_name in ("kettle", "rice-cooker", "microwave"):
            metrics = test_one_appliance("ukdale", app_name, method, info, config)
            table.add_row(*[str(metric) for metric in metrics])
        for app_name in ("furnace", "washer-dryer", "microwave"):
            metrics = test_one_appliance("redd", app_name, method, info, config)
            table.add_row(*[str(metric) for metric in metrics])
    elif dataset == "ukdale" and app == "all":
        # test all appliances in ukdale
        for app_name in ("kettle", "rice-cooker", "microwave"):
            metrics = test_one_appliance("ukdale", app_name, method, info, config)
            table.add_row(*[str(metric) for metric in metrics])
    elif dataset == "redd" and app == "all":
        # test all appliances in redd
        for app_name in ("furnace", "washer-dryer", "microwave"):
            metrics = test_one_appliance("redd", app_name, method, info, config)
            table.add_row(*[str(metric) for metric in metrics])
    else:
        # test specific appliance in specific dataset
        metrics = test_one_appliance(dataset, app, method, info, config)
        table.add_row(*[str(metric) for metric in metrics])
    console = Console(record=True)
    console.print(table)


if __name__ == "__main__":
    test()
