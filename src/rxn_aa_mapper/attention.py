"""Attention handling and transformations

Attentions are always calculated from the tokenized SMILES and enzymes. To convert this into a proper atom mapping software,
the need arises to map the tokens (which include parentheses, special tokens, and bonds) to the atom domain.

This module contains all of the helper methods needed to convert the attention matrix into the atom domain,
separating on reactants and products, including special tokens and not including special tokens, in the atom
domain / in the token domain, and accounting for adjacent atoms in molecules.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from numpy.core.fromnumeric import nonzero

from .rxn_utils import (
    get_atom_types_smiles,
    get_mask_for_tokens,
    number_tokens,
    tokens_to_adjacency,
)


class AttentionScorer:
    def __init__(
        self,
        rxn: str,
        tokens: List[str],
        attentions: np.ndarray,
        special_tokens: List[str] = ["[CLS]", "[SEP]"],
        attention_multiplier: float = 90.0,
        mask_mapped_reactant_atoms: bool = True,
        output_attentions: bool = False,
        smiles_aa_sequence_separator: str = "|",
        top_k: int = 1,
    ):
        """Convenience wrapper for mapping attentions into the atom domain, separated by reactants and products, and introducing neighborhood locality.

        Args:
            rxn: reactants + enzyme + products
            tokens: Tokenized reaction of length N
            attentions: NxN attention matrix
            special_tokens: Special tokens used by the model that do not count as atoms
            attention_multiplier: Amount to increase the attention connection from adjacent atoms of a newly mapped product atom to adjacent atoms of the newly mapped reactant atom.
                Boosts the likelihood of an atom having the same adjacent atoms in reactants and products
            mask_mapped_reactant_atoms: If true, zero attentions to reactant atoms that have already been mapped
            output_attentions: If true, output the raw attentions along with generated atom maps
            smiles_aa_sequence_separator: token used to separate reactants to enzyme in the reaction format
            top_k: number of aa_sequence's token to map to each reactant's atom
        """
        if tokens[0] == "[CLS]":
            tokens.pop(0)
        if tokens[-1] == "[SEP]":
            tokens.pop()

        self.rxn, self.tokens, self.attentions, self.special_tokens, self.N = (
            rxn,
            tokens,
            attentions,
            special_tokens,
            len(tokens),
        )
        self.attention_multiplier = attention_multiplier
        self.mask_mapped_reactant_atoms = mask_mapped_reactant_atoms
        self.output_attentions = output_attentions
        self.smiles_aa_sequence_separator = smiles_aa_sequence_separator
        self.top_k = top_k
        # get reactant and aa_sequence indices
        try:
            self.split_ind = self.tokens.index(self.smiles_aa_sequence_separator)
            self.reactant_inds, self.aa_sequence_inds = (
                slice(0, self.split_ind),
                slice(self.split_ind + 1, self.N),
            )
            self.reactant_tokens = self.tokens[self.reactant_inds]
            self.aa_sequence_tokens = self.tokens[self.aa_sequence_inds]
        except ValueError:
            raise ValueError(
                f"The reaction {self.rxn} is not completed, cannot find the symbol | which separates the reactants from enzyme"
            )

        # Mask of reactant's atoms
        self.reactant_atom_token_mask = np.array(
            get_mask_for_tokens(self.reactant_tokens, self.special_tokens)
        ).astype(np.bool_)

        # Reactant's atom numbered in np.array
        self.reactant_token2atom = np.array(number_tokens(self.reactant_tokens))
        self.reactant_atom2token = {
            k: v
            for k, v in zip(
                self.reactant_token2atom, range(len(self.reactant_token2atom))
            )
        }

        # adjacency graph for reactant's atoms
        self.reactant_adjacency_matrix = tokens_to_adjacency(
            self.reactant_tokens
        ).astype(np.bool_)

        self._rxp_filt_atoms = None
        self._pxr_filt_atoms = None
        self._rnums_atoms: Optional[np.ndarray] = None
        self._nreactant_atoms: Optional[int] = None
        self._precursors_atom_types: Optional[List[int]] = None
        self._nproduct_atoms: Optional[int] = None

        self.attention_multiplier_matrix = np.ones_like(
            self.combined_attentions_filt_atoms
        ).astype(float)

    @property
    def atom_attentions(self):
        """The MxM attention matrix, selected for only attentions that are from reactant atoms, to atoms"""
        return (
            self.attentions[self.reactant_atom_token_mask]
            .T[self.reactant_atom_token_mask]
            .T
        )

    @property
    def adjacent_atom_attentions(self):
        """The MxM attention matrix, where all attentions are zeroed if the attention is not to an adjacent atom."""
        atts = self.atom_attentions.copy()
        mask = np.logical_not(self.reactant_adjacency_matrix)
        atts[mask] = 0
        return atts

    @property
    def adjacency_matrix_reactants(self):
        """The adjacency matrix of the reactants"""
        return self.reactant_adjacency_matrix

    @property
    def rxp(self):
        """Subset of attentions relating the reactants to the enzyme"""
        i = self.split_ind
        return self.attentions[:i, (i + 1) :]

    @property
    def rxp_filt_atoms(self):
        """RXP only the atoms, no special tokens"""
        if self._rxp_filt_atoms is None:
            self._rxp_filt_atoms = self.rxp[[i != -1 for i in self.rnums]][
                :, [i != -1 for i in self.pnums]
            ]
        return self._rxp_filt_atoms

    @property
    def pxr(self):
        """Subset of attentions relating the products to the reactants"""
        i = self.split_ind
        return self.attentions[(i + 1) :, :i]

    @property
    def pxr_filt_atoms(self):
        """PXR only the atoms, no special tokens"""
        if self._pxr_filt_atoms is None:
            self._pxr_filt_atoms = self.pxr[[i != -1 for i in self.pnums]][
                :, [i != -1 for i in self.rnums]
            ]
        return self._pxr_filt_atoms

    @property
    def rnums(self):
        """Get atom indexes for the reactant tokens.

        Numbers in this vector that are >= 0 are atoms, whereas indexes == -1 represent special tokens (e.g., bonds, parens, [CLS])
        """
        return self.reactant_token2atom

    @property
    def pnums(self):
        """Get atom indexes for just the product tokens.

        Numbers in this vector that are >= 0 are atoms, whereas indexes == -1 represent special tokens (e.g., bonds, parens, [CLS])
        """
        return list(range(len(self.aa_sequence_tokens)))

    @property
    def rnums_atoms(self):
        """Get atom indexes for the reactant ATOMS, without the [CLS].

        Numbers in this vector that are >= 0 are atoms, whereas indexes == -1 represent special tokens (e.g., bonds, parens, [CLS])
        """
        if self._rnums_atoms is None:
            self._rnums_atoms = np.array([a for a in self.rnums if a != -1])
        return self._rnums_atoms

    @property
    def pnums_atoms(self):
        """Get atom indexes for just the product ATOMS, without the [SEP].

        Numbers in this vector that are >= 0 are atoms, whereas indexes == -1 represent special tokens (e.g., bonds, parens, [CLS])
        """
        return self.pnums

    @property
    def nreactant_atoms(self):
        """The number of atoms in the reactants"""
        if self._nreactant_atoms is None:
            self._nreactant_atoms = len(self.rnums_atoms)

        return self._nreactant_atoms

    @property
    def nproduct_atoms(self):
        """The number of atoms in the product"""
        if self._nproduct_atoms is None:
            self._nproduct_atoms = len(self.aa_sequence_tokens)
        return self._nproduct_atoms

    @property
    def rtokens(self):
        """Just the reactant tokens"""
        return self.tokens[self.reactant_inds]

    @property
    def ptokens(self):
        """Just the product tokens"""
        return self.tokens[self.aa_sequence_inds]

    @property
    def combined_attentions(self):
        """Summed pxr and rxp"""
        return self.rxp + self.pxr.T

    @property
    def combined_attentions_filt_atoms(self):
        """Summed pxr_filt_atoms and rxp_filt_atoms (no special tokens, no "non-atom" tokens)"""
        return self.rxp_filt_atoms + self.pxr_filt_atoms.T

    def get_neighboring_attentions(self, atom_num) -> np.ndarray:
        """Get a vector of shape (n_atoms,) representing the neighboring attentions to an atom number.
        Non-zero attentions are the attentions for neighboring atoms
        """
        return self.atom_attentions[atom_num] * self.reactant_adjacency_matrix[atom_num]

    def get_neighboring_atoms(self, atom_num):
        """Get the atom indexes neighboring the desired atom"""
        return nonzero(np.atleast_1d(self.reactant_adjacency_matrix[atom_num]))[0]

    def get_precursors_atom_types(self):
        """Convert reactants into their atomic numbers"""
        if self._precursors_atom_types is None:
            self._precursors_atom_types = get_atom_types_smiles("".join(self.rtokens))
        return self._precursors_atom_types

    def _get_combined_normalized_attentions(self):
        """ Get normalized attention matrix from product atoms to candidate reactant atoms. """
        combined_attention = np.multiply(
            self.combined_attentions_filt_atoms, self.attention_multiplier_matrix
        )
        row_sums = combined_attention.sum(axis=1)
        normalized_attentions = np.divide(
            combined_attention,
            row_sums[:, np.newaxis],
            out=np.zeros_like(combined_attention),
            where=row_sums[:, np.newaxis] != 0,
        )
        return normalized_attentions

    def generate_attention_guided_rxp_atom_mapping(
        self, absolute_aa_sequence_inds: bool = False
    ):
        """
        Generate attention guided AA sequencce to reactant atom mapping.

        Args:
            absolute_aa_sequence_inds: If True, adjust all indexes related to the product to be relative to that atom's position
                in the entire reaction SMILES.
        """
        rxp_mapping_vector: Dict[int, Any] = {
            i: -1 for i in range(len(self.reactant_tokens))
        }
        output = {}
        confidences = np.ones(len(self.reactant_tokens))
        mapping_tuples = []

        attention_matrix = self._get_combined_normalized_attentions()
        output["rxppxr_attns"] = attention_matrix

        for i in range(len(self.reactant_atom_token_mask)):
            reactant_atom_to_map = int(np.argmax(np.max(attention_matrix, axis=1)))
            row_scores = attention_matrix[reactant_atom_to_map]
            aa_sequence_token_indices = row_scores.argsort()[-self.top_k :][::-1]

            rxp_mapping_vector[reactant_atom_to_map] = []
            inert_confidence = np.ones(self.top_k)

            for i, aa_token_index in enumerate(aa_sequence_token_indices):
                confidence = attention_matrix[reactant_atom_to_map, aa_token_index]
                if np.isclose(confidence, 0.0):
                    confidence = 1.0
                inert_confidence[i] = confidence
                rxp_mapping_vector[reactant_atom_to_map].append(aa_token_index)

            confidences[reactant_atom_to_map] = np.prod(inert_confidence)

            self._update_attention_multiplier_matrix(
                int(aa_sequence_token_indices[-1]), int(reactant_atom_to_map)
            )

            shift_index_aa_sequence_token = (
                self.nreactant_atoms if absolute_aa_sequence_inds else 0
            )
            for i, aa_sequence_token_index in enumerate(aa_sequence_token_indices):
                mapping_tuples.append(
                    (
                        reactant_atom_to_map,
                        shift_index_aa_sequence_token + aa_sequence_token_index,
                        inert_confidence[i],
                    )
                )

            attention_matrix = self._get_combined_normalized_attentions()

        output["rxp_mapping_vector"] = rxp_mapping_vector
        output["confidences"] = confidence
        output["mapping_tuples"] = mapping_tuples
        return output

    def _update_attention_multiplier_matrix(
        self, product_atom: int, reactant_atom: int
    ):
        """Perform the "neighbor multiplier" step of the atom mapping

        Increase the attention connection between the neighbors of specified product atom
        to the neighbors of the specified reactant atom. A stateful operation.
        Args:
            product_atom: Atom index of the product atom (relative to the beginning of the products)
            reactant_atom: Atom index of the reactant atom (relative to the beginning of the reactants)
        """
        if not reactant_atom == -1:
            neighbors_in_products = np.arange(self.nproduct_atoms)
            neighbors_in_reactants = self.adjacency_matrix_reactants[reactant_atom]

            self.attention_multiplier_matrix[
                np.ix_(neighbors_in_reactants, neighbors_in_products)
            ] *= float(self.attention_multiplier)

            if self.mask_mapped_reactant_atoms:
                self.attention_multiplier_matrix[reactant_atom] = np.zeros(
                    self.nproduct_atoms
                )

    def __len__(self):
        """Length of provided tokens"""
        return len(self.tokens)

    def __repr__(self):
        return f"AttentionScorer(`{self.rxn[:50]}...`)"
