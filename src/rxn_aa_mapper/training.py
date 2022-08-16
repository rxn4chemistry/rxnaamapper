"""Training utilities."""
import os
import subprocess
from typing import Any, Dict, Union

import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from .dataset import (
    DATASETS,
    EnzymaticReactionDataset,
    EnzymaticReactionLightningDataModule,
)
from .model import EnzymaticReactionLightningModule

# CCC Specific Code --------------------------------------------------------------------------------------------------------------------------
def fix_infiniband():
    ibv = subprocess.run("ibv_devinfo", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = ibv.stdout.decode("utf-8").split("\n")
    exclude = ""
    for line in lines:
        if "hca_id:" in line:
            name = line.split(":")[1].strip()
        if "\tport:" in line:
            port = line.split(":")[1].strip()
        if "link_layer:" in line and "Ethernet" in line:
            exclude = exclude + f"{name}:{port},"

    if exclude:
        exclude = "^" + exclude[:-1]
        os.environ["NCCL_IB_HCA"] = exclude


def set_env():
    # print("Using " + str(torch.cuda.device_count()) + " GPUs---------------------------------------------------------------------")

    LSB_MCPU_HOSTS = os.environ["LSB_MCPU_HOSTS"].split(
        " "
    )  # Parses Node list set by LSF, in format hostname proceeded by number of cores requested
    HOST_LIST = LSB_MCPU_HOSTS[::2]  # Strips the cores per node items in the list
    LSB_JOBID = os.environ[
        "LSB_JOBID"
    ]  # Parses Node list set by LSF, in format hostname proceeded by number of cores requested
    os.environ["MASTER_ADDR"] = HOST_LIST[
        0
    ]  # Sets the MasterNode to thefirst node on the list of hosts
    os.environ["MASTER_PORT"] = "5" + LSB_JOBID[-5:-1]
    os.environ["NODE_RANK"] = str(
        HOST_LIST.index(os.environ["HOSTNAME"])
    )  # Uses the list index for node rank, master node rank must be 0
    os.environ[
        "NCCL_SOCKET_IFNAME"
    ] = "ib,bond"  # avoids using docker of loopback interface
    os.environ[
        "NCCL_DEBUG"
    ] = "INFO"  # sets NCCL debug to info, during distributed training, bugs in code show up as nccl errors
    os.environ["NCCL_IB_CUDA_SUPPORT"] = "1"  # Force use of infiniband
    os.environ["NCCL_TOPO_DUMP_FILE"] = "NCCL_TOP.%h.xml"
    os.environ["NCCL_DEBUG_FILE"] = "NCCL_DEBUG.%h.%p.txt"
    # print(os.environ["HOSTNAME"] + " MASTER_ADDR: " + os.environ["MASTER_ADDR"])
    # print(os.environ["HOSTNAME"] + " MASTER_PORT: " + os.environ["MASTER_PORT"])
    # print(os.environ["HOSTNAME"] + " NODE_RANK " + os.environ["NODE_RANK"])
    # print(os.environ["HOSTNAME"] + " NCCL_SOCKET_IFNAME: " + os.environ["NCCL_SOCKET_IFNAME"])
    # print(os.environ["HOSTNAME"] + " NCCL_DEBUG: " + os.environ["NCCL_DEBUG"])
    # print(os.environ["HOSTNAME"] + " NCCL_IB_CUDA_SUPPORT: " + os.environ["NCCL_IB_CUDA_SUPPORT"])
    # print("Using " + str(torch.cuda.device_count()) + " GPUs---------------------------------------------------------------------")
    # print(os.environ["HOSTNAME"] + " LSB_MCPU_HOSTS: " + os.environ["LSB_MCPU_HOSTS"])
    # print("Using " + str(torch.cuda.device_count()) + " GPUs---------------------------------------------------------------------")
    # print(os.environ["HOSTNAME"] + " HOST_LIST: ")
    # print(HOST_LIST)
    # print("Using " + str(torch.cuda.device_count()) + " GPUs---------------------------------------------------------------------")
    # print(os.environ["HOSTNAME"] + " HOSTNAME: " + os.environ["HOSTNAME"])
    # print("Using " + str(torch.cuda.device_count()) + " GPUs---------------------------------------------------------------------")


# --------------------------------------------------------------------------------------------------------------------------------------------


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

    # CCC Specific Code --------------------------------------------------------------------------------------------------------------------------
    fix_infiniband()
    set_env()
    # --------------------------------------------------------------------------------------------------------------------------------------------

    data_module = get_data_module(dataset_args)
    model_architecture["vocab_size"] = data_module.train_dataset.tokenizer.vocab_size
    model = EnzymaticReactionLightningModule(model_args, model_architecture)
    model.model.resize_token_embeddings(model_architecture["vocab_size"])

    log_dir = trainer_args["log_dir"]
    os.makedirs(log_dir, exist_ok=True)

    del trainer_args["log_dir"]
    lightning_logger = TensorBoardLogger(
        name="logger",
        save_dir=log_dir,
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
