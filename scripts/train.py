import argparse
from pathlib import Path
from datetime import datetime

from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint
from ignite.metrics import Average, Accuracy
import torch
import torch.nn as nn
from torch.utils import tensorboard

from semantic_segmentation_ros.dataset import SegDataset
from semantic_segmentation_ros.networks import get_network

def main():
    args = parse_args()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}

    # Create log directory for tensorboard
    time_stamp = datetime.now().strftime("%y-%m-%d-%H-%M")
    description = f"{time_stamp},batch_size={args.batch_size},lr={args.lr}"
    logdir = args.logdir / description 

    # Create data loaders
    train_loader, val_loader = create_train_val_loaders(
        args.dataset, args.batch_size, args.val_split, kwargs
    )

    # Build the network
    net = get_network(args.net).to(device)

    # Define optimizer and metrics
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    metrics = {
        "loss" : Average(lambda out: out[3]),
    }

    # Create ignite engines for training and validation
    trainer = create_trainer(net, optimizer, loss_fn, metrics, device)
    evaluator = create_evaluator(net, loss_fn, metrics, device)

    # Log training progress to the terminal and tensorboard
    ProgressBar(persist=True, ascii=True).attach(trainer)
    train_writer, val_writer = create_summary_writers(net, device, logdir)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_train_results(engine):
        epoch, metrics = trainer.state.epoch, trainer.state.metrics
        train_writer.add_scalar("loss", metrics["loss"], epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        epoch, metrics = trainer.state.epoch, evaluator.state.metrics
        val_writer.add_scalar("loss", metrics["loss"], epoch)

    # Checkpoint model
    checkpoint_handler = Checkpoint(
        {args.net: net},
        str(logdir),
        "semantic_segmentation_ros",
        n_saved=100,
        global_step_transform=lambda *_: trainer.state.epoch,
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler)

    # Run the training loop
    trainer.run(train_loader, max_epochs=args.epochs)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--logdir", type=Path, default="log/training")
    parser.add_argument("--net", default="unet")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    return parser.parse_args()

def create_train_val_loaders(dir, batch_size, val_split, kwargs):
    dataset = SegDataset(dir)
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs
    )
    return train_loader, val_loader

def loss_fn(y_pred, y):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(y_pred, y)
    return loss

def prepare_batch(batch, device):
    images, masks = batch
    images = images.float()
    images = images.to(device)
    masks = masks.to(device)
    return images, masks


def create_trainer(net, optimizer, loss_fn, metrics, device):
    def _update(_, batch):
        net.train()
        optimizer.zero_grad()

        x, y = prepare_batch(batch, device)
        y_pred = net(x)
        loss = loss_fn(y_pred, y)

        loss.backward()
        optimizer.step()

        return x, y_pred, y, loss

    trainer = Engine(_update)

    for name, metric in metrics.items():
        metric.attach(trainer, name)

    return trainer

def create_evaluator(net, loss_fn, metrics, device):
    def _inference(_, batch):
        net.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device)
            y_pred = net(x)
            loss = loss_fn(y_pred, y)
        return x, y_pred, y, loss

    evaluator = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator

def create_summary_writers(net, device, log_dir):
    train_path = log_dir / "train"
    val_path = log_dir / "validation"
    train_writer = tensorboard.SummaryWriter(train_path, flush_secs=60)
    val_writer = tensorboard.SummaryWriter(val_path, flush_secs=60)
    return train_writer, val_writer

if __name__ == "__main__":
    main()