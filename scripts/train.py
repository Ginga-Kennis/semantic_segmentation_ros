import argparse
from pathlib import Path
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils import tensorboard

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers.tqdm_logger import ProgressBar
from ignite.metrics import Loss
from ignite.handlers import Checkpoint
from ignite.handlers.tensorboard_logger import *

from semantic_segmentation_ros.dataset import SegDataset
from semantic_segmentation_ros.networks import get_model

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Log directory
    time_stamp = datetime.now().strftime("%m%d%H%M")
    description = f"{time_stamp},batch_size={args.batch_size},lr={args.lr}"
    logdir = args.logdir / description 

    # Build the network
    model = get_model(name=args.model).to(device)

    # Create data loaders
    train_loader, val_loader = create_train_val_loaders(datadir=args.datadir, val_split=args.val_split, batch_size=args.batch_size)

    # Define optimizer and criterion(loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Create ignite engines for training and validation
    trainer = create_supervised_trainer(model, optimizer, criterion, device)

    metrics = {
        "loss": Loss(criterion)
    }
    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    val_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

    # Log training progress to the terminal and tensorboard
    ProgressBar(persist=False).attach(trainer)
    train_writer, val_writer = create_summary_writers(logdir)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_train_results(engine):
        train_evaluator.run(train_loader)
        epoch, metrics = trainer.state.epoch, train_evaluator.state.metrics
        train_writer.add_scalar("loss", metrics["loss"], epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        val_evaluator.run(val_loader)
        epoch, metrics = trainer.state.epoch, val_evaluator.state.metrics
        val_writer.add_scalar("loss", metrics["loss"], epoch)


    # Checkpoint model
    checkpoint_handler = Checkpoint(
        {args.model: model},
        str(logdir),
        "",
        n_saved=100,
        global_step_transform=lambda *_: trainer.state.epoch,
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler)

    # run training
    trainer.run(train_loader, max_epochs=args.epochs)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=Path, default="data")
    parser.add_argument("--logdir", type=Path, default="log/training")
    parser.add_argument("--model", type=str, default="unet")
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=500)
    return parser.parse_args()

def create_train_val_loaders(datadir, val_split, batch_size):
    dataset = SegDataset(datadir)
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def create_summary_writers(log_dir):
    train_path = log_dir / "train"
    val_path = log_dir / "validation"
    train_writer = tensorboard.SummaryWriter(train_path, flush_secs=60)
    val_writer = tensorboard.SummaryWriter(val_path, flush_secs=60)
    return train_writer, val_writer

if __name__ == "__main__":
    main()