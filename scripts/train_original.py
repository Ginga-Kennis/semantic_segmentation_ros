import argparse
from pathlib import Path
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils import tensorboard

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.metrics import Loss
from ignite.handlers import Checkpoint
from ignite.contrib.handlers.tensorboard_logger import *

from semantic_segmentation_ros.dataset import SegDataset
from semantic_segmentation_ros.networks import get_network

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Log directory
    time_stamp = datetime.now().strftime("%m%d%H%M")
    description = f"{time_stamp},batch_size={args.batch_size},lr={args.lr}"
    logdir = args.logdir / description 

    # Build the network
    model = get_network(name=args.net).to(device)

    # Create data loaders
    train_loader, val_loader = create_train_val_loaders(val_split=args.val_split, batch_size=args.batch_size)

    # Define optimizer and criterion(loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Create ignite engines for training and validation
    trainer = create_supervised_trainer(model, optimizer, criterion, device)

    val_metrics = {
        "loss": Loss(criterion)
    }
    evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)

    # Log training progress to the terminal and tensorboard
    ProgressBar(persist=True, ascii=True).attach(trainer)
    tb_logger = TensorboardLogger(log_dir=logdir)

    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=10),
        tag="training",
        output_transform=lambda loss: {"batch_loss": loss},
    )


    # Checkpoint model
    checkpoint_handler = Checkpoint(
        {args.net: model},
        str(logdir),
        "",
        n_saved=100,
        global_step_transform=lambda *_: trainer.state.epoch,
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler)

    
    trainer.run(train_loader, max_epochs=args.epochs)





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=Path, default="log/training")
    parser.add_argument("--net", type=str, default="unet")
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    return parser.parse_args()

def create_train_val_loaders(val_split, batch_size):
    dataset = SegDataset("./data/")
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

if __name__ == "__main__":
    main()