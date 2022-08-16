"""Tokenization utilties for exrepssions."""


import re
from typing import Callable, List, Optional

from tokenizers import Tokenizer
from transformers import AlbertTokenizer, BertTokenizer

SMILES_TOKENIZER_PATTERN = r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"

AA_SEQUENCE_TOKENIZERS_LOADERS = {
    "albert": lambda tokenizer_filepath: AlbertTokenizer.from_pretrained(
        tokenizer_filepath, do_lower_case=False
    ),
    "bert": lambda tokenizer_filepath: BertTokenizer.from_pretrained(
        tokenizer_filepath, do_lower_case=False
    ),
    "generic": lambda tokenizer_filepath: Tokenizer.from_file(tokenizer_filepath),
}

AA_SEQUENCE_TOKENIZER_FUNCTIONS = {
    "albert": lambda text, tokenizer: tokenizer.tokenize(" ".join(list(text))),
    "bert": lambda text, tokenizer: tokenizer.tokenize(" ".join(list(text))),
    "generic": lambda text, tokenizer: tokenizer.encode(text).tokens,
}


class RegexTokenizer:
    """Run regex tokenization"""

    def __init__(self, regex_pattern: str, suffix: str = "") -> None:
        """Constructs a RegexTokenizer.

        Args:
            regex_pattern: regex pattern used for tokenization.
            suffix: optional suffix for the tokens. Defaults to "".
        """
        self.regex_pattern = regex_pattern
        self.regex = re.compile(self.regex_pattern)
        self.suffix = suffix

    def tokenize(self, text: str) -> List[str]:
        """Regex tokenization.

        Args:
            text: text to tokenize.

        Returns:
            extracted tokens.
        """
        tokens = [f"{token}{self.suffix}" for token in self.regex.findall(text)]
        return tokens


class AASequenceTokenizer:
    """Run AA sequence tokenization."""

    def __init__(
        self, tokenizer_filepath: str, tokenizer_type: str = "generic"
    ) -> None:
        """
        Constructs an AASequenceTokenizer.

        Args:
            tokenizer_filepath: path to a serialized AA sequence tokenizer.
            tokenizer_type: type of tokenization to use. Defaults to "generic".
        """
        self.tokenizer_filepath = tokenizer_filepath
        self.tokenizer_type = tokenizer_type
        self.tokenizer = AA_SEQUENCE_TOKENIZERS_LOADERS.get(
            self.tokenizer_type, AA_SEQUENCE_TOKENIZERS_LOADERS["generic"]
        )(self.tokenizer_filepath)
        self.tokenizer_fn = AA_SEQUENCE_TOKENIZER_FUNCTIONS.get(
            self.tokenizer_type, AA_SEQUENCE_TOKENIZER_FUNCTIONS["generic"]
        )

    def tokenize(self, text: str) -> List[str]:
        """Tokenization of a property.

        Args:
            text: text to tokenize.

        Returns:
            extracted tokens.
        """
        return self.tokenizer_fn(text, self.tokenizer)


class EnzymaticReactionTokenizer:
    """Constructs a EnzymaticReactionTokenizer using AA sequence."""

    def __init__(
        self,
        aa_sequence_tokenizer_filepath: Optional[str] = None,
        smiles_aa_sequence_separator: str = "|",
        reaction_separator: str = ">>",
        aa_sequence_tokenizer_type: str = "generic",
    ) -> None:
        """Constructs an EnzymaticReactionTokenizer.

        Args:
            aa_sequence_tokenizer_filepath: file to a serialized AA sequence tokenizer.
            smiles_aa_sequence_separator: separator between reactants and AA sequence. Defaults to "|".
            reaction_separator: reaction sides separator. Defaults to ">>".
            aa_sequence_tokenizer_type: type of tokenization to use for aa sequences. Defaults to "generic".
        """
        # define tokenization utilities
        self.smiles_tokenizer = RegexTokenizer(
            regex_pattern=SMILES_TOKENIZER_PATTERN, suffix="_"
        )
        self.aa_sequence_tokenizer_filepath = aa_sequence_tokenizer_filepath
        self.aa_sequence_tokenizer_type = aa_sequence_tokenizer_type
        self.aa_sequence_tokenizer = self._get_aa_tokenizer_fn()
        self.smiles_aa_sequence_separator = smiles_aa_sequence_separator
        self.reaction_separator = reaction_separator

    def tokenize(self, text: str) -> List[str]:
        """Tokenize a text representing an enzymatic reaction with AA sequence information.

        Args:
            text: text to tokenize.

        Returns:
            extracted tokens.
        """
        product = ""
        aa_sequence = ""
        try:
            reactants_and_aa_sequence, product = text.split(self.reaction_separator)
        except ValueError:
            reactants_and_aa_sequence = text
        try:
            reactants, aa_sequence = reactants_and_aa_sequence.split(
                self.smiles_aa_sequence_separator
            )
        except ValueError:
            reactants = reactants_and_aa_sequence
        tokens = []
        tokens.extend(self.smiles_tokenizer.tokenize(reactants))
        if aa_sequence:
            tokens.append(self.smiles_aa_sequence_separator)
            tokens.extend(self.aa_sequence_tokenizer(aa_sequence))
        if product:
            tokens.append(self.reaction_separator)
            tokens.extend(self.smiles_tokenizer.tokenize(product))
        return tokens

    def _get_aa_tokenizer_fn(self) -> Callable:
        """Definition of the tokenizer for the aa sequence
        Returns:
            a callable function
        """
        if self.aa_sequence_tokenizer_filepath is not None:
            fn = AASequenceTokenizer(
                tokenizer_filepath=self.aa_sequence_tokenizer_filepath,
                tokenizer_type=self.aa_sequence_tokenizer_type,
            ).tokenize
            return fn
        else:
            return list


class LMEnzymaticReactionTokenizer(BertTokenizer):
    """
    Constructs a EnzymaticReactionBertTokenizer.
    Adapted from https://github.com/huggingface/transformers

    Args:
        vocabulary_file: path to a token per line vocabulary file.
    """

    def __init__(
        self,
        vocabulary_file: str,
        aa_sequence_tokenizer_filepath: Optional[str] = None,
        aa_sequence_tokenizer_type: str = "generic",
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        smiles_aa_sequence_separator: str = "|",
        reaction_separator: str = ">>",
        **kwargs,
    ) -> None:
        """Constructs an ExpressionTokenizer.

        Args:
            vocabulary_file: vocabulary file containing tokens.
            aa_sequence_tokenizer_filepath: file to a serialized AA sequence tokenizer.
            aa_sequence_tokenizer_type: type of tokenization to use for aa sequences. Defaults to "generic".
            unk_token: unknown token. Defaults to "[UNK]".
            sep_token: separator token. Defaults to "[SEP]".
            pad_token: pad token. Defaults to "[PAD]".
            cls_token: cls token. Defaults to "[CLS]".
            mask_token: mask token. Defaults to "[MASK]".
            smiles_aa_sequence_separator: separator between reactants and AA sequence. Defaults to "|".
            reaction_separator: reaction sides separator. Defaults to ">>".
        """
        super().__init__(
            vocab_file=vocabulary_file,
            do_lower_case=False,
            do_basic_tokenize=True,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )
        # define tokenization utilities
        self.tokenizer = EnzymaticReactionTokenizer(
            aa_sequence_tokenizer_filepath=aa_sequence_tokenizer_filepath,
            smiles_aa_sequence_separator=smiles_aa_sequence_separator,
            reaction_separator=reaction_separator,
            aa_sequence_tokenizer_type=aa_sequence_tokenizer_type,
        )

    @property
    def vocab_list(self) -> List[str]:
        """List vocabulary tokens.

        Returns:
            a list of vocabulary tokens.
        """
        return list(self.vocab.keys())

    def _tokenize(self, text: str) -> List[str]:  # type:ignore
        """Tokenize a text representing an enzymatic reaction with AA sequence information.

        Args:
            text: text to tokenize.

        Returns:
            extracted tokens.
        """
        return self.tokenizer.tokenize(text)
