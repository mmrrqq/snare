import os
from pathlib import Path

import hydra
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import numpy as np
import random

import torch
import models
from data.dataset import CLIPGraspingDataset
from torch.utils.data import DataLoader


@hydra.main(config_path="cfgs", config_name="train", version_base="1.1")
def main(cfg):    
    # set random seeds
    seed = cfg['train']['random_seed']
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_float32_matmul_precision('high')

    hydra_dir = Path(os.getcwd())
    checkpoint_path = hydra_dir / 'checkpoints'
    last_checkpoint_path = os.path.join(checkpoint_path, 'last.ckpt')
    last_checkpoint = last_checkpoint_path \
        if os.path.exists(last_checkpoint_path) and cfg['train']['load_from_last_ckpt'] else None

    checkpoint_callback = ModelCheckpoint(
        monitor=cfg['wandb']['saver']['monitor'],
        dirpath=checkpoint_path,
        filename='{epoch:04d}-{val_acc:.5f}',
        save_top_k=1,
        save_last=True,
    )
    logger = TensorBoardLogger(hydra_dir)
    trainer = Trainer(
        devices=[0],
        fast_dev_run=cfg['debug'],
        callbacks=[checkpoint_callback],
        max_epochs=cfg['train']['max_epochs'],
        logger=logger,
        log_every_n_steps=1
    )

    # dataset
    train = CLIPGraspingDataset(cfg, mode='train')
    valid = CLIPGraspingDataset(cfg, mode='valid', lang_feats=train.lang_feats, img_feats=train.img_feats)
    test = CLIPGraspingDataset(cfg, mode='test', lang_feats=train.lang_feats, img_feats=train.img_feats)

    # model
    model = models.names[cfg['train']['model']](cfg, train, valid, freeze_mapping_layer=False)

    # resume epoch and global_steps
    if last_checkpoint and cfg['train']['load_from_last_ckpt']:
        print(f"Resuming: {last_checkpoint}")
        last_ckpt = torch.load(last_checkpoint)
        trainer.fit_loop.epoch_progress.current.completed = last_ckpt['epoch']
        trainer.global_step = last_ckpt['global_step']
        del last_ckpt
    
    trainer.fit(
        model,
        train_dataloaders=DataLoader(train, batch_size=cfg['train']['batch_size'], num_workers=16),
        val_dataloaders=DataLoader(valid, batch_size=cfg['train']['batch_size'], num_workers=16),
    )

    trainer.test(
        dataloaders=DataLoader(test, batch_size=cfg['train']['batch_size'], num_workers=16),
        ckpt_path='best'
    )

if __name__ == "__main__":
    main()
