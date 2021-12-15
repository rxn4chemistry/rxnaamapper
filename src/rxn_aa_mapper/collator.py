"""Data collator for n-gram masked language modeling"""
from typing import Optional, Tuple

import torch
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase


class NGramDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        mlm: bool = True,
        mlm_probability: float = 0.15,
        pad_to_multiple_of: Optional[int] = None,
        n_gram: int = 3,
    ):
        """Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they are not all of the same length.
        Args:
            tokenizer: The tokenizer used for encoding the data.
            mlm: Whether or not to use masked language modeling.
            mlm_probablity: The probability with which to (randomly) mask tokens in the input
            pad_to_multiple_of: If set will pad the sequence to a multiple of the provided value.
            n_gram: number of continguous tokens to mask
        """
        super().__init__(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability)
        self.n_gram = n_gram
        self.index_smiles_aa_sequence_separator = self.tokenizer.vocab[  # type:ignore
            "|"
        ]
        self.index_reaction_separator = self.tokenizer.vocab[">>"]  # type:ignore

    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 1000% MASK,
        Args:
            inputs: batch of input_ids
            special_tokens_mask: special tokens to avoid mask

        Returns:
            a tuple of masked input_ids and labels.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(  # type:ignore
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # blockify the masked tokens
        for batch_idx in range(labels.shape[0]):
            low_index = (
                inputs[batch_idx] == self.index_smiles_aa_sequence_separator
            ).nonzero()
            upper_index = (inputs[batch_idx] == self.index_reaction_separator).nonzero()
            if low_index.shape[0] != 0 and upper_index.shape[0] != 0:
                low, up = low_index[0][0], upper_index[0][0]
                mask_sum = masked_indices[batch_idx][low : (up + 1)].sum()
                masked_indices[batch_idx][low : (up + 1)] = False
                num_blocs = (mask_sum + self.n_gram - 1) // self.n_gram
                if num_blocs == 0:
                    continue
                bloc_indices = torch.randint(low, up + 1, (num_blocs,))
                for bloc_idx in bloc_indices:
                    low_bloc_idx = max(low, bloc_idx - self.n_gram // 2)
                    up_bloc_idx = min(up, bloc_idx + self.n_gram // 2)
                    masked_indices[batch_idx][low_bloc_idx : (up_bloc_idx + 1)] = True

        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(  # type:ignore
            self.tokenizer.mask_token
        )

        return inputs, labels
