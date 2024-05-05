import argparse
import json
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils import tensorboard

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers.tqdm_logger import ProgressBar
from ignite.metrics import Loss
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.handlers.tensorboard_logger import *

from semantic_segmentation_ros.dataset import SegDataset
from semantic_segmentation_ros.metrics import MeanIoU
from semantic_segmentation_ros.networks import get_model

def main(config: dict) -> None:
    """
    Main function that handles setup to training, validation, and model saving.

    Inputs:
        config (dict): Configuration settings loaded from a config file.
    Outputs:
        None (The model is trained and results are logged.)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create log directory
    time_stamp = datetime.now().strftime("%m-%d-%H-%M")
    description = f'{time_stamp},model={config["arch"]["model_name"]},batch_size={config["dataloader"]["batch_size"]},lr={config["train"]["lr"]}/'
    logdir = config["log"]["path"] + description 

    # Create data loaders
    train_loader, val_loader = create_train_val_loaders(**config["dataset"], **config["dataloader"])

    # Build the network
    model = get_model(**config["arch"]).to(device)

    # Define optimizer, criterion and metrics
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["lr"], weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    metrics = {
        "loss": Loss(criterion),
        "mIOU": MeanIoU(num_classes=config["arch"]["classes"], device=device)
    }

    # Create ignite engines for training and validation
    trainer = create_supervised_trainer(model, optimizer, criterion, device)
    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    val_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

    # Log training progress to the terminal and tensorboard
    ProgressBar(persist=False).attach(trainer)
    train_writer, val_writer = create_summary_writers(logdir)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_train_results(engine: Engine) -> None:
        train_evaluator.run(train_loader)
        epoch, metrics = trainer.state.epoch, train_evaluator.state.metrics
        train_writer.add_scalar("loss", metrics["loss"], epoch)
        train_writer.add_scalar("mIOU", metrics["mIOU"], epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine: Engine) -> None:
        val_evaluator.run(val_loader)
        epoch, metrics = trainer.state.epoch, val_evaluator.state.metrics
        val_writer.add_scalar("loss", metrics["loss"], epoch)
        val_writer.add_scalar("mIOU", metrics["mIOU"], epoch)

    # Checkpoint best model
    model_checkpoint = ModelCheckpoint(
        dirname=str(logdir),
        score_function=lambda engine: engine.state.metrics['mIOU'],
        score_name="mIOU",
        n_saved=5,
        create_dir=True,
        global_step_transform=global_step_from_engine(trainer), 
    )
    val_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {config["arch"]["model_name"]: model})

    # Run the training loop
    trainer.run(train_loader, max_epochs=config["train"]["epochs"])

def create_train_val_loaders(path: str, labels: list, augmentations: dict, batch_size: int, num_workers: int, pin_memory: bool) -> tuple:
    """
    Generate loaders for training and validation datasets.

    Inputs:
        path (str): Path to the image data.
        labels (list): List of labels.
        augmentations (dict): Augmentations to be applied as preprocessing.
        batch_size (int): Batch size.
        num_workers (int): Number of workers for the data loader.
        pin_memory (bool): Whether to use pin memory.

    Outputs:
        train_loader (DataLoader), val_loader (DataLoader): Loaders for training and validation datasets.
    """
    train_dataset = SegDataset(path + "/train", labels, augmentations, is_train=True)
    val_dataset = SegDataset(path + "/val", labels, augmentations, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader

def create_summary_writers(log_dir: str) -> tuple:
    """
    Create TensorBoard summary writers for logging.

    Inputs:
        log_dir (str): Path to the log directory.
    Outputs:
        Tuple containing the training and validation SummaryWriters.
    """
    train_path = log_dir + "train"
    val_path = log_dir + "validation"
    
    train_writer = tensorboard.SummaryWriter(train_path, flush_secs=60)
    val_writer = tensorboard.SummaryWriter(val_path, flush_secs=60)
    return train_writer, val_writer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='config/train.json', help="Path to config JSON file")
    args = parser.parse_args()
    
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    return config

if __name__ == "__main__":
    config = parse_args()
    main(config)