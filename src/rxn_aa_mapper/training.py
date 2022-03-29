"""Training utilities."""
import os
from typing import Any, Dict, Union

import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from .dataset import (
    DATASETS,
    EnzymaticReactionDataset,
    EnzymaticReactionLightningDataModule,
)
from .model import EnzymaticReactionLightningModule


def get_data_module(
    dataset_args: Dict[str, Union[float, str, int]],
) -> EnzymaticReactionLightningDataModule:
    """
    Get a data module for enzymatic reactions.
    Args:
        dataset_args: dictionary containing all the necessary parameters for the dataset creation.
    Returns:
        data module for enzymatic reactions.
    """
    return EnzymaticReactionLightningDataModule(
        dataset_args,
        DATASETS.get(
            str(dataset_args.get("dataset_type", "enzymatic")), EnzymaticReactionDataset
        ),
    )


def train(
    model_args: Dict[str, Union[float, str, int]],
    model_architecture: Dict[str, Union[float, str, int]],
    dataset_args: Dict[str, Union[float, str, int]],
    trainer_args: Dict[str, Any],
) -> None:
    """
    Train a model.
    Args:
        model_args: dictionary containing all the parameters for the mode configuration.
        model_architecture: dictionary containing the information related to the architecture of the model.
        dataset_args: dictionary containing all the necessary parameters for the dataset creation.
        training_args: dictionary containing all the necessary parameters for the training routine.
    """
    data_module = get_data_module(dataset_args)
    model_architecture["vocab_size"] = data_module.train_dataset.tokenizer.vocab_size
<<<<<<< HEAD
    model = EnzymaticReactionLightningModule(model_args)
    model.model.resize_token_embeddings(model_architecture["vocab_size"])
=======

    model = EnzymaticReactionLightningModule(
        model_args, model_architecture, from_albert=from_albert, from_bert=from_bert
    )
>>>>>>> 310cf4d7edbe43c717c01b7afb95b7c1e3176674

    log_dir = trainer_args["log_dir"]
    os.makedirs(log_dir, exist_ok=True)

    del trainer_args["log_dir"]
    lightning_logger = WandbLogger(
        name="mlm-logger", save_dir=log_dir, log_model=True, project="rxn-aa-mapper"
    )
    trainer_args["logger"] = lightning_logger
    if not torch.cuda.is_available():
        del trainer_args["gpus"]

    if not isinstance(trainer_args["val_check_interval"], int):
        trainer_args["val_check_interval"] = 10000
        logger.warning(
            f"please set trainer['val_check_interval'] to an integer value, defaulting to {trainer_args['val_check_interval']}"
        )
    if (
        "accelerator" not in trainer_args
        or trainer_args.get("accelerator", "ddp") == "ddp_spawn"
    ):
        trainer_args["accelerator"] = "ddp"
        logger.warning(
            f"ddp_spawn not supported because of pickle issues, defaulting to {trainer_args['accelerator']}"
        )

    # gather the callbacks
    trainer_args["callbacks"] = []
    if "early_stopping_callback" in trainer_args:
        callback: Callback = EarlyStopping(**trainer_args["early_stopping_callback"])
        del trainer_args["early_stopping_callback"]
        trainer_args["callbacks"].append(callback)

    if "model_checkpoint_callback" in trainer_args:
        callback = ModelCheckpoint(**trainer_args["model_checkpoint_callback"])
        del trainer_args["model_checkpoint_callback"]
        trainer_args["callbacks"].append(callback)

    trainer = pl.Trainer(**trainer_args)
    trainer.fit(model, data_module)


def checkpoint_to_module(
    input_checkpoint: str,
    model_args: Dict[str, Union[float, str, int]],
    model_architecture: Dict[str, Union[float, str, int]],
) -> EnzymaticReactionLightningModule:
    """
    Transform a checkpoint into a module.
    Args:
        input_checkpoint: model checkpoint.
        model_args: dictionary containing all the parameters for the mode configuration.
        model_architecture: dictionary containing the information related to the architecture of the model.
    Returns:
        the ligthining module.
    """
    return EnzymaticReactionLightningModule.load_from_checkpoint(
        checkpoint_path=input_checkpoint,
        model_args=model_args,
        model_architecture=model_architecture,
    )
