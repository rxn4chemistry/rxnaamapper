"""Model for Masked Language Modeling."""
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch.optim as optim
from loguru import logger
from torch import Tensor
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM
)


class EnzymaticReactionLightningModule(pl.LightningModule):
    """Pytorch lightning model for MLM training on enzymatic reactions."""

    def __init__(
        self,
        model_args: Dict[str, Any],
        model_architecture: Dict[str, Any],
        from_albert: Optional[bool] = False,
        from_bert: Optional[bool] = False,
    ) -> None:
        """
        Construct an EnzymaticReaction lightning module.

        Args:
            model_args: a dictionary object containing all the necessary arguments for the model creation.
            model_architecture: a dictionary containing all the information related to the architecture of the model.
        """
        super().__init__()
        self.model_args = model_args
        resume = self.model_args.get("resume-training", False)
        if resume:
            logger.info(
                f"resume the training from the checkpoint : {self.model_args['model']}"
            )
            self.model = AutoModelForMaskedLM.from_pretrained(model_args["model"])
        else:
            if from_albert:
                self.config = AutoConfig.from_pretrained(
                    "Rostlab/prot_albert", **model_architecture
                )

            elif from_bert:
                self.config = AutoConfig.from_pretrained(
                    "Rostlab/prot_bert", **model_architecture
                )

            else:
                self.config = AutoConfig.from_pretrained(
                    model_args["model"], **model_architecture
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
