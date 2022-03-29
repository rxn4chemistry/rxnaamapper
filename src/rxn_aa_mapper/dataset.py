"""Dataset routines --filtering, dataset building"""
import fnmatch
import os
import random
from collections import namedtuple
from typing import Any, Dict, Iterator, List, Optional, Type, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from loguru import logger
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling

from .collator import NGramDataCollatorForLanguageModeling
from .tokenization import LMEnzymaticReactionTokenizer


class EnzymaticReactionDataset(IterableDataset):
    """torch-like dataset based textual representation of enzymatic reactions."""

    column_names = ["reaction", "EC", "source"]
    iterator_and_filepath = namedtuple(
        "iterator_and_filepath", ["iterator", "filepath"]
    )

    def __init__(self, dataset_args: Dict[str, Any], stage: str = "train") -> None:
        """
        Build dataset from the csv file of enzymatic reaction.

        Args:
            dataset_args: dictionary containing the folder where the csv file of reaction are stored.
            stage: string that indicates if we on training or validation stage.

        """
        aa_sequence_tokenizer_filepath = dataset_args.get(
            "aa_sequence_tokenizer_filepath", None
        )

        self.tokenizer = LMEnzymaticReactionTokenizer(
            dataset_args["vocabulary_file"], aa_sequence_tokenizer_filepath
        )

        if stage == "train":
            dataset_dir = dataset_args["train_dataset_dir"]
        else:
            dataset_dir = dataset_args["val_dataset_dir"]

        self.aa_reaction_smiles_filepaths = [
            os.path.join(dataset_dir, entry)
            for entry in os.listdir(dataset_dir)
            if fnmatch.fnmatch(entry, dataset_args["pattern"])
        ]

        self.max_length_token = dataset_args["max_length_token"]
        self.shuffle = dataset_args["shuffle"]
        self.seed = dataset_args["seed"]
        self.dataset_args = dataset_args

        if self.seed is None:
            self.seed = random.randint(0, 42)
        self.reset_seed(self.seed)

    @classmethod
    def reset_seed(cls, seed: int) -> None:
        """Reset seed for random number generation."""
        random.seed(seed)
        np.random.seed(seed)

    def filepaths_fn(self, filepaths: List[str]) -> List[str]:
        if self.shuffle:
            return random.sample(filepaths, len(filepaths))
        else:
            return filepaths

    def preprocess_chunk_fn(self, chunk: pd.DataFrame) -> pd.DataFrame:
        if self.shuffle:
            return chunk.sample(frac=1)
        else:
            return chunk

    def lazy_generator(self, filepaths: List[str]):
        """
        Create a generator which will load the file lazily.
        Yields:
            iterator of length `len(filepaths) * self.dataset_args["chunk_size"]` containing a stream of the inputs suitable to a BERT-based model
        """
        dataframe_and_filepath_list = [
            self.get_dataframe_iterator(filepath)
            for filepath in self.filepaths_fn(filepaths)
        ]
        dataframe_and_filepath_list = [
            item for item in dataframe_and_filepath_list if item
        ]
        is_list_none = False
        while True:
            samples = [
                self.get_next_sample_from_iterator(index, dataframe_and_filepath_list)
                for index in range(len(dataframe_and_filepath_list))
            ]
            none_list = list(map(lambda x: x is None, samples))
            if all(none_list):
                if is_list_none:
                    logger.info("End of the dataset")
                    return
                else:
                    is_list_none = True
                    continue
            samples_df = pd.concat(samples, axis=0)
            samples_df = self.preprocess_chunk_fn(samples_df)
            samples_df = self.filter_tokenized_reactions_by_length(
                samples_df, self.tokenizer, self.max_length_token
            )
            if len(samples_df) == 0:
                continue
            for _, row in samples_df.iterrows():
                yield self.tokenizer(
                    row["reaction"],
                    max_length=self.max_length_token,
                    padding="max_length",
                    return_special_tokens_mask=True,
                )

    def get_dataframe_iterator(self, filepath: str) -> Optional[Any]:
        """Get the iterator of a dataframe
        Args:
            filepath: path of the csv file

        Returns:
            an iterator on a dataframe
        """

        return self.iterator_and_filepath(
            pd.read_csv(
                filepath,
                names=self.column_names,
                header=None,
                chunksize=self.dataset_args["chunk_size"],
                engine="python",
            ),
            filepath,
        )

        try:
            return self.iterator_and_filepath(
                pd.read_csv(
                    filepath,
                    names=self.column_names,
                    header=None,
                    chunksize=self.dataset_args["chunk_size"],
                    engine="python",
                ),
                filepath,
            )
        except Exception:
            logger.warning(f"error building generator for the file: {filepath}")
            return None

    def get_next_sample_from_iterator(
        self, index: int, dataframe_and_filepath_list: List[Any]
    ) -> Optional[pd.DataFrame]:
        """generate sample from multiple data sources
        Args:
            index: index of the iterator to apply the `next` operation
            dataframe_and_filepath_list: list of the tuple made up of iterator and the filepath where the data is drew.

        Returns:
            a merged dataframe
        """
        try:
            return next(dataframe_and_filepath_list[index].iterator)
        except StopIteration:
            dataframe_and_filepath_list[index] = self.iterator_and_filepath(
                pd.read_csv(
                    dataframe_and_filepath_list[index].filepath,
                    names=self.column_names,
                    header=None,
                    chunksize=self.dataset_args["chunk_size"],
                    engine="python",
                ),
                dataframe_and_filepath_list[index].filepath,
            )
            return None

    def __iter__(self) -> Iterator:
        """
        Get an item from the dataset.

        Returns:
            dict of tensor containing the input_ids, token_type_ids and the attention_mask.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # single worker case, a single process generates all data.
            return self.lazy_generator(self.aa_reaction_smiles_filepaths)
        else:
            logger.info(
                "multiple workers loading the dataset offsetting seeds by worker ids"
            )
            self.seed = self.seed + worker_info.id
            self.reset_seed(int(self.seed))
            return self.lazy_generator(self.aa_reaction_smiles_filepaths)

    @classmethod
    def filter_tokenized_reactions_by_length(
        cls,
        data: pd.DataFrame,
        tokenizer: LMEnzymaticReactionTokenizer,
        max_length_token: int,
    ) -> pd.DataFrame:
        """Filter rtokenized reactions by length.

        Args:
            data : dataframe containing the enzymatic-reaction, the dataframe data should have at
                least the column reaction which contains the textual representation of the reaction.
            tokenizer: tokenizer inherited from the BertTokenizer.
            max_length_token: maximum length in tokens. Defaults to 512.

        Returns:
            dataframe derived from the input dataframe by discarding the reactions with more
            than max_length_token tokens according to the tokenizer provided as argument
        """
        data["length_token"] = data["reaction"].apply(
            lambda raw: len(tokenizer.tokenize(raw)) <= max_length_token - 3
        )
        data = data[data["length_token"]]
        del data["length_token"]
        return data

    @classmethod
    def filter_and_save_tokenized_reactions_by_length(
        cls,
        tokenizer: LMEnzymaticReactionTokenizer,
        input_folder: str,
        output_folder: str,
        max_length_token: int = 512,
    ) -> None:
        """
        Filter and save tokenized reactions by length.

        Args:
            tokenizer: enzymatic reaction tokenizer inherits from BertTokenizer.
            input_folder: absolute path of the folder which contains the reactions for each enzymatic class.
            output_folder: absolute path of the folder that will contain the filtered enzymatic reactions.
            max_length_token: maximum length in tokens. Defaults to 512.
        """

        for filename in tqdm(os.listdir(input_folder)):
            data = pd.read_csv(
                os.path.join(input_folder, filename), names=cls.column_names
            )
            data = cls.filter_tokenized_reactions_by_length(
                data, tokenizer, max_length_token
            )
            if len(data) == 0:
                pass
            data.to_csv(os.path.join(output_folder, filename))


class EnzymaticReactionMTLDataset(EnzymaticReactionDataset):
    """Dataset for multi-task transfer learning."""

    def __init__(self, dataset_args: Dict[str, Any], stage: str = "train") -> None:
        """
        Build dataset from the csv files of enzymatic and organic reactions.

        Args:
            dataset_args: dictionary containing the folder where the csv file of reaction are stored.
            stage: string that indicates if we on training or validation stage.
        """
        super().__init__(dataset_args=dataset_args, stage=stage)
        if stage == "train":
            dataset_dir = dataset_args["train_organic_dataset_dir"]
        else:
            dataset_dir = dataset_args["val_organic_dataset_dir"]
        self.organic_reaction_smiles_filepaths = [
            os.path.join(dataset_dir, entry)
            for entry in os.listdir(dataset_dir)
            if fnmatch.fnmatch(entry, dataset_args["pattern"])
        ]
        organic_reaction_weight = dataset_args.get("organic_dataset_weight", 0.9)
        enzymatic_reaction_weight = dataset_args.get("enzymatic_dataset_weight", 0.1)
        self.sampling_frame = dataset_args.get("sampling_frame", 100)
        total_weight = organic_reaction_weight + enzymatic_reaction_weight
        self.organic_reaction_weight = organic_reaction_weight / total_weight
        self.enzymatic_reaction_weight = enzymatic_reaction_weight / total_weight

    def weighted_generator(self):
        """
        Create a generator which will load the file lazily.

        Yields:
            dict of tensors containing the input_ids, token_type_ids and the attention_mask.
        """
        enzymatic_lazy_generator = self.lazy_generator(
            self.aa_reaction_smiles_filepaths
        )
        organic_reaction_lazy_generator = self.lazy_generator(
            self.organic_reaction_smiles_filepaths
        )
        while True:
            num_enzymatic_reactions = int(
                self.sampling_frame * self.enzymatic_reaction_weight
            )
            num_organic_reactions = int(
                self.sampling_frame * self.organic_reaction_weight
            )
            samples = []
            for _ in range(num_enzymatic_reactions):
                try:
                    samples.append(next(enzymatic_lazy_generator))
                except StopIteration:
                    pass
            for _ in range(num_organic_reactions):
                try:
                    samples.append(next(organic_reaction_lazy_generator))
                except StopIteration:
                    pass
            np.random.shuffle(samples)
            yield from iter(samples)

    def __iter__(self) -> Iterator:
        """
        Get an item from the dataset.

        Returns:
            dict of tensor containing the input_ids, token_type_ids and the attention_mask.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # single worker case, a single process generates all data.
            return self.weighted_generator()
        else:
            logger.info(
                "multiple workers loading the dataset offsetting seeds by worker ids"
            )
            self.seed = self.seed + worker_info.id
            self.reset_seed(int(self.seed))
            return self.weighted_generator()


DATASETS = {
    "enzymatic": EnzymaticReactionDataset,
    "enzymatic-organic": EnzymaticReactionMTLDataset,
}

COLLATORS = {
    "sparse": DataCollatorForLanguageModeling,
    "n-gram": NGramDataCollatorForLanguageModeling,
}


class EnzymaticReactionLightningDataModule(pl.LightningDataModule):
    """Pytorch-lightning-style data module for enzymatic reaction dataset."""

    def __init__(
        self,
        dataset_args: Dict[str, Union[float, str, int]],
        dataset_class: Type[
            Union[EnzymaticReactionDataset, EnzymaticReactionMTLDataset]
        ] = EnzymaticReactionDataset,
    ) -> None:
        """
        Initialize the data module.

        Args:
            dataset_args: dictionary containing the metadata for the lightning data module creation.
            dataset_class: class of the dataset to be used in the data module. Defaults to EnzymaticReactionDataset.
        """
        super().__init__()
        self.dataset_args = dataset_args
        self.dataset_class = dataset_class
        self.setup_datasets()

    def setup_datasets(self) -> None:
        """
        Setup data module.

        Split the main dataset into training and validation set according to the factors provided
        in dataset_args at the initialization of the data module.
        """
        self.train_dataset = self.dataset_class(self.dataset_args, stage="train")
        self.val_dataset = self.dataset_class(self.dataset_args, stage="val")
        self.collator_class = COLLATORS.get(
            str(self.dataset_args.get("collator-type", "sparse")),
            DataCollatorForLanguageModeling,
        )
        self.mlm_data_collator = self.collator_class(
            tokenizer=self.train_dataset.tokenizer,
            mlm_probability=self.dataset_args["mlm_probability"],
        )

    def train_dataloader(self) -> DataLoader:
        """
        Create the DataLoader for the traning step.

        Returns:
            pytorch-like dataloader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.dataset_args["batch_size"]),
            num_workers=int(self.dataset_args["num_dataloader_workers"]),
            collate_fn=self.mlm_data_collator,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create the DataLoader for the traning step.

        Returns:
            pytorch-like dataloader.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=int(self.dataset_args["batch_size"]),
            num_workers=int(self.dataset_args["num_dataloader_workers"]),
            collate_fn=self.mlm_data_collator,
        )
