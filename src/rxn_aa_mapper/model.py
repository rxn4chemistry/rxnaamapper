"""Model for Masked Language Modeling."""
from typing import Any, Dict, List, cast

import pytorch_lightning as pl
import torch.optim as optim
from loguru import logger
from torch import Tensor
from transformers import (
    AlbertConfig,
    AlbertForMaskedLM,
    AutoConfig,
    AutoModelForMaskedLM,
    BertConfig,
    BertForMaskedLM,
)


class ModelLoader:
    """Laod an Albert model"""

    def __init__(
        self,
        model_name: str = None,
        model_class: List[Any] = None,
        model_architecture: Dict[str, Any] = None,
    ) -> None:
        """
        Based on the path, load the model.

        Args:
            model_name: a string containing the name of the model.
            model_class: a list containing the transformer classes to call.
            model_architecture: a dictionary containing all the information related to the architecture of the model.
        """

        self.model_name = model_name
        self.model_architecture = model_architecture
        model_class = cast(List[Any], model_class)
        self.configuration = model_class[0]
        self.MaskedLM = model_class[1]

    def get_model(self) -> None:
        """Give back the model.

        Returns:
            a Hugginface model.
        """
        self.config = self.configuration.from_pretrained(
            self.model_name, **self.model_architecture
        )

        model = self.MaskedLM(self.config)

        return model


MODEL_CLASS = {
    "albert": [AlbertConfig, AlbertForMaskedLM],
    "bert": [BertConfig, BertForMaskedLM],
}


class EnzymaticReactionLightningModule(pl.LightningModule):
    """Pytorch lightning model for MLM training on enzymatic reactions."""

    def __init__(
        self,
        model_args: Dict[str, Any],
    ) -> None:
        """
        Construct an EnzymaticReaction lightning module.

        Args:
            model_args: a dictionary object containing all the necessary arguments for the model creation.
        """
        super().__init__()
        self.model_args = model_args
        self.model_architecture = model_args.get("architecture", {})

        resume = self.model_args.get("resume-training", False)
        if resume:
            logger.info(
                f"resume the training from the checkpoint : {self.model_args['model']}"
            )
            self.model = AutoModelForMaskedLM.from_pretrained(model_args["model"])
        else:
            if "albert" in model_args["model"].lower():
                self.model = ModelLoader(
                    model_args["model"],
                    MODEL_CLASS.get("albert"),
                    self.model_architecture,
                )
                self.model = self.model.get_model()

            elif (
                "albert" not in model_args["model"].lower()
                and "bert" in model_args["model"].lower()
            ):
                self.model = ModelLoader(
                    model_args["model"],
                    MODEL_CLASS.get("bert"),
                    self.model_architecture,
                )
                self.model = self.model.get_model()

            else:
                self.config = AutoConfig.from_pretrained(
                    model_args["model"], **self.model_architecture
                )
                self.model = AutoModelForMaskedLM.from_config(self.config)

    def forward(self, x: Tensor) -> Tensor:  # type:ignore
        """
        Forward pass on Transformer model

        Args:
            x: tensor of shape (batch_size, seq_length) containing the input_ids.

        Returns:
            logits of the model.
        """
        return self.model(x).logits

    def configure_optimizers(
        self,
    ) -> Dict[str, Any]:
        """Create and return the optimizer.

        Returns:
            output (dict of str: Any):
                - optimizer: the optimizer used to update the parameter
                - scheduler: the scheduler used to reduce the learning rate on plateau
                - monitor: the metric that the scheduler will track over the training.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.model_args["lr"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "min",
            factor=self.model_args["lr_decay"],
            patience=self.model_args["patience"],
        )
        output = {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
        return output

    def training_step(  # type:ignore
        self, batch: Dict[str, Tensor], batch_idx: int
    ) -> Tensor:
        """
        Training step which encompasses the forward pass and the computation of the loss value.

        Args:
            batch: dictionary containing the input_ids and optionally the token_type_ids and the attention_type.
            batch_idx: index of the current batch, unused.

        Returns:
            loss computed on the batch.
        """
        loss = self.model(**batch).loss
        self.log("train_loss", loss)
        return loss

    def validation_step(  # type:ignore
        self, batch: Dict[str, Tensor], batch_idx: int
    ) -> Tensor:
        """
        Validation step which encompasses the forward pass and the computation of the loss value.

        Args:
            batch: dictionary containing the input_ids and optionally the token_type_ids and the attention_type.
            batch_idx: index of the current batch, unused.

        Returns:
            loss computed on the batch.
        """
        loss = self.model(**batch).loss
        self.log("val_loss", loss)
        return loss
