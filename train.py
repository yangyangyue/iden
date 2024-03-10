import argparse
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger as Logger
from sconf import Config

from gen_net import Gmime, GmimeConfig
from read_data import get_loader


def train_one_appliance(set_name: str, app_name: str, model_name: str):
    """
    Train and validation for one appliance.

    Args:
        set_name: the name of dataset, i.e., UKDALE or REDD.
        app_name: the name of appliance to be trained.
        model_name: the name of model.
    """
    # checkpoint path
    ckpt_path = Path("weights") / model_name / f"{set_name}_{app_name}.pth"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    # where to save the fault case
    case_dir = Path("case") / model_name / f"{set_name}_{app_name}"
    case_dir.mkdir(parents=True, exist_ok=True)

    # build train loader and val loader
    if set_name == "ukdale":
        houses = ["house_1", "house_5"]
        house_ratios = [0.15, 0.1]
    else:
        houses = ["house_2", "house_3", "house_4", "house_5", "house_6"]
        house_ratios = [1, 1, 1, 1, 1]
    train_loader, val_loader = get_loader(
        set_name, houses, house_ratios, app_name, config.batch_size
    )

    # build model
    model_config = GmimeConfig(
        config.n_class,
        config.hidden_dimension,
        config.max_length,
        config.lr,
        config.warmup_steps,
        config.min_lr,
        config.gamma,
        config.lr_step,
    )
    model = Gmime(model_config)

    # training init
    lr_callback = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        monitor="f1",
        dirpath=ckpt_path,
        filename="best-model",
        save_top_k=1,
        mode="max",
    )
    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        max_epochs=config.max_epochs,
        gradient_clip_val=config.gradient_clip_val,
        log_every_n_steps=20,
        # precision="bf16-mixed",
        logger=Logger("train", project="Gmame", config=dict(config)),
        callbacks=[lr_callback, checkpoint_callback],
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = Config(Path("config") / "train.yaml")
    train_one_appliance("ukdale", "kettle", "gmame")
