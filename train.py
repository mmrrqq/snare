import os
from pathlib import Path

import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import random

import torch
import models
from data.dataset import CLIPGraspingDataset
from torch.utils.data import DataLoader


@hydra.main(config_path="cfgs", config_name="train")
def main(cfg):    
    # set random seeds
    seed = cfg['train']['random_seed']
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

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
    trainer = Trainer(
        devices=[0],
        fast_dev_run=cfg['debug'],
        callbacks=[checkpoint_callback],
        max_epochs=cfg['train']['max_epochs']        
    )

    # dataset
    train = CLIPGraspingDataset(cfg, mode='train')
    valid = CLIPGraspingDataset(cfg, mode='valid')
    test = CLIPGraspingDataset(cfg, mode='test')

    # model
    model = models.names[cfg['train']['model']](cfg, train, valid, freeze_mapping_layer=False)

    # resume epoch and global_steps
    if last_checkpoint and cfg['train']['load_from_last_ckpt']:
        print(f"Resuming: {last_checkpoint}")
        last_ckpt = torch.load(last_checkpoint)
        trainer.current_epoch = last_ckpt['epoch']
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
