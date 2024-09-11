"""Core RXN Attention Mapper module."""
import re
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO
from loguru import logger
from rdkit import Chem
from transformers import AutoConfig, AutoModel

from .attention import AttentionScorer
from .rxn_utils import (
    AA_DICT,
    INTERACTION_PATTERN,
    REACTION_SEPARATOR,
    SMILES_AA_SEQUENCE_SEPARATOR,
    NotCanonicalizableSmilesException,
    NotReactionException,
    decode_aa_sequence_indices,
    generate_atom_mapped_reaction_atoms,
    process_reaction,
)
from .tokenization import LMEnzymaticReactionTokenizer


class RXNAAMapper:
    """Wrap the Transformer model, corresponding tokenizer, and attention scoring algorithms.

    Performs mapping between:
        reactant atoms and enzyme tokens
        reactant atoms and product one
        product atoms and enzyme tokens
        (reactant + enzyme complex) and product
    """

    def __init__(
        self,
        config: Dict[str, Any] = {},
    ):
        """
        RXNAAMapper constructor.
        Args:
            config: Config dict, leave it empty to have the official rxnmapper.
        """

        self.model_path = config["model_path"]
        self.attention_multiplier = config.get("attention_multiplier", 90.0)
        self.head = config.get("head", 2)
        self.layers = config.get("layers", [11])
        self.logger = logger
        self.top_k = config.get("top_k", 3)
        self.vocabulary_file = config["vocabulary_file"]
        self.aa_sequence_tokenizer_filepath = config.get(
            "aa_sequence_tokenizer_filepath", None
        )
        self.aa_sequence_tokenizer_type = config.get(
            "aa_sequence_tokenizer_type", "generic"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model_tokenizer()

    def _load_model_tokenizer(self) -> None:
        """
        Load transformer and tokenizer model.
        """
        self.tokenizer = LMEnzymaticReactionTokenizer(
            self.vocabulary_file,
            self.aa_sequence_tokenizer_filepath,
            self.aa_sequence_tokenizer_type,
        )
        self.config_model = AutoConfig.from_pretrained(
            self.model_path,
            output_attentions=True,
            output_hidden_states=False,
            output_past=False,
        )
        self.model = AutoModel.from_pretrained(
            self.model_path, config=self.config_model
        ).eval()
        self.model = self.model.to(self.device)

    def convert_batch_to_attns(
        self,
        rxn_list: List[str],
        force_layer: Optional[int] = None,
        force_head: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """Extract desired attentions from a given batch of reactions.
        Args:
            rxn_list: List of reactions to mape
            force_layer: If given, override the default layer used for RXNMapper
            force_head: If given, override the default head used for RXNMapper

        Returns:
            attentions: list of n-by-n matrix representing the attention mapping for each rxn reaction in the batch
        """
        if force_layer is None:
            use_layers: Any = self.layers
        else:
            use_layers = [force_layer]

        if force_head is None:
            use_head = self.head
        else:
            use_head = force_head

        inputs_model = self.tokenizer.batch_encode_plus(
            rxn_list, padding=True, return_tensors="pt"
        )
        inputs_model = {k: v.to(self.device) for k, v in inputs_model.items()}

        with torch.no_grad():
            attentions = self.model(**inputs_model).attentions
        attentions = torch.cat(
            [a.unsqueeze(1) for i, a in enumerate(attentions) if i in use_layers],
            dim=1,
        )
        attention_mask = inputs_model["attention_mask"].to(torch.bool)
        attentions = attentions[:, :, use_head, :, :].mean(dim=[1])
        attentions = [a[mask][:, mask] for a, mask in zip(attentions, attention_mask)]
        return attentions

    def get_reactant_left_hand_side_attention(
        self,
        rxn_list: List[str],
        force_layer: Optional[int] = None,
        force_head: Optional[int] = None,
        rxn_separator: str = ">>",
    ) -> List[torch.Tensor]:
        """
        Extract the reaction-reagent token mapping from the full token mapping matrix
        Args:
            rxn_list: List of reactions to map
            force_layer: If given, override the default layer used for RXNMapper
            force_head: If given, override the default head used for RXNMapper
            reaction_separator: token used to separate reactants and enzymne to the products
        Returns:
            attentions: list of n-by-n matrix representing the reactant-reagent attention mapping for each rxn reaction in the batch
        """
        attentions = self.convert_batch_to_attns(
            rxn_list, force_layer=force_layer, force_head=force_head
        )
        output = []
        for rxn, attn in zip(rxn_list, attentions):
            try:
                precursors, _ = rxn.split(rxn_separator)
                reaction_end_index = len(self.tokenizer.tokenize(precursors))
                # do not extract the attention for the [CLS] token
                output.append(
                    attn[1 : reaction_end_index + 1, 1 : reaction_end_index + 1]
                )
            except Exception:
                raise NotReactionException(
                    f"{rxn}. Unpected reaction format, the expected format is `reactants|aa_sequence>>products`"
                )
        return output

    def get_reactant_aa_sequence_attention_guided_maps(
        self,
        rxns: List[str],
        zero_set_r: bool = True,
        canonicalize_rxns: bool = True,
        detailed_output: bool = False,
        absolute_aa_sequence_inds: bool = False,
        force_layer: Optional[int] = None,
        force_head: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Generate atom-mapping for reactions.
        Args:
            rxns: List of reaction SMILES (no reactant/enzyme split)
            zero_set_r: Mask mapped reactant atoms (default: True)
            canonicalize_rxns: Canonicalize reactions (default: True)
            detailed_output: Get back more information (default: False)
            absolute_enzyme_inds: Different atom indexing (default: False)
            force_layer: Force specific layer (default: None)
            force_head: Force specific head (default: None)
        Returns:
            Mapped reactions with confidence score (List):
                - mapped_rxn: Mapped reaction SMARTS
                - confidence: Model confidence in the mapping rxn
            `detailed_output=True` additionally outputs...
                - rxp_mapping_vector: Vector used to generate the product atom indexes for the mapping
                - rxp_confidences: The corresponding confidence for each atom's map
                - mapping_tuples: (product_atom_index (relative to first product atom), corresponding_reactant_atom, confidence)
                - rxppxr_attns: Just the attentions from the product tokens to the reactant tokens
                - tokensxtokens_attns: Full attentions for all tokens
                - tokens: Tokens that were inputted into the model
        """
        results = []
        if canonicalize_rxns:
            rxns = [process_reaction(rxn) for rxn in rxns]
        attns = self.get_reactant_left_hand_side_attention(
            rxns,
            force_layer=force_layer,
            force_head=force_head,
        )
        for attn, rxn in zip(attns, rxns):
            precursors, _ = rxn.split(REACTION_SEPARATOR)
            just_tokens = [
                token.replace("_", "") for token in self.tokenizer.tokenize(precursors)
            ]
            attn_matrix = attn.cpu().numpy()
            attention_scorer = AttentionScorer(
                rxn,
                just_tokens,
                attn_matrix,
                attention_multiplier=float(self.attention_multiplier),
                mask_mapped_reactant_atoms=zero_set_r,
                output_attentions=detailed_output,
                top_k=int(self.top_k),
            )
            output = attention_scorer.generate_attention_guided_rxp_atom_mapping(
                absolute_aa_sequence_inds=absolute_aa_sequence_inds
            )

            result = {
                "mapped_rxn": generate_atom_mapped_reaction_atoms(
                    rxn, output["rxp_mapping_vector"]
                )[0],
                "confidence": np.prod(output["confidences"]),
            }

            if detailed_output:
                result["rxp_mapping_vector"] = output["rxp_mapping_vector"]
                result["rxp_confidences"] = output["confidences"]
                result["mapping_tuples"] = output["mapping_tuples"]
                result["rxppxr_attns"] = output["rxppxr_attns"]
                result["tokensxtokens_attn"] = attn
                result["tokens"] = just_tokens

            results.append(result)
        return results

    def get_overlap_and_penalty_score(
        self,
        predicted_tokens: List[Tuple[int, int]],
        ground_truth_token_indices: List[Tuple[int, int]],
        aa_sequence: str,
    ) -> Dict[str, Any]:
        """Computer the overlapping score between two lists of interval.
        Args:
            predicted_tokens: boundaries of the tokens predicted as the active sites of the enzyme.
            ground_truth_tokens: boundaries of the tokens experimentally determined as the active sites of the enzyme.
            aa_sequence: amino acid sequence.

        Returns:
            a dictionary the overlap score, the fpr and other metrics.
        """
        lst_pred = []
        for i, j in predicted_tokens:
            lst_pred.extend([k for k in range(i, j)])

        lst_pred = set(lst_pred)

        lst_gt = []
        for i, j in ground_truth_token_indices:
            lst_gt.extend([k for k in range(i, j)])

        lst_gt = set(lst_gt)

        amino = [i for i in range(len(aa_sequence))]

        TP = len(lst_pred.intersection(lst_gt))
        FP = len(lst_pred.difference(lst_gt))
        TN = len(set(amino).difference((lst_gt).union(lst_pred)))

        if (FP + TP) != len(lst_pred):
            print("there is a mistake in the calculations")

        output = {
            "overlap_score": TP / len(lst_gt),
            "false_positive_rate": FP / (FP + TN),
        }

        return output 

    def get_predicted_active_site(
        self,
        mapped_rxn: Any,
        rxn_separator: str = REACTION_SEPARATOR,
        smiles_aa_sequence_separator: str = SMILES_AA_SEQUENCE_SEPARATOR,
    ):
        """Get the predicted active site from the mapped reaction
        Args:
            mapped_rxn: the mapped reaction
            rxn_separator: token used to separate reactants and enzyme from products
            smiles_aa_sequence_separator: token used to separate smiles enzyme

        Returns:
            a tuple containing the overlap score the penalty score and the false positive rate
        """
        reactant_mols = None
        if not isinstance(mapped_rxn, str):
            reactant_mols = mapped_rxn["reactants_mol"]
            mapped_rxn = mapped_rxn["rxn"]

        sites: Dict[int, int] = defaultdict(int)
        reactants = ""
        try:
            reactants, _ = mapped_rxn.split(rxn_separator)
        except Exception:
            reactants = mapped_rxn
        precursor_tokens = self.tokenizer.tokenize(reactants)
        try:
            reactants, _ = reactants.split(smiles_aa_sequence_separator)
        except Exception:
            raise ValueError(
                "The reaction should have at least the reactant and the enzyme"
            )

        if reactant_mols is None:
            reactant_mols = Chem.MolFromSmiles(reactants)
            if not reactant_mols:
                raise NotCanonicalizableSmilesException("Molecule not canonicalizable")

        for atom in reactant_mols.GetAtoms():
            if atom.HasProp("molAtomMapNumber"):
                list_aa_sequence_token_index = decode_aa_sequence_indices(
                    int(atom.GetProp("molAtomMapNumber"))
                )
                for aa_sequence_token_index in list_aa_sequence_token_index:
                    sites[aa_sequence_token_index] += 1
        reactant_length = len(list(reactant_mols.GetAtoms()))

        filtered_sites = {
            k: v
            for k, v in sorted(sites.items(), key=lambda item: item[1])[
                -reactant_length:
            ]
        }
        if len(filtered_sites) == 0:
            raise ValueError(
                "the frequencies of the active sites do not meet the threshold requirement"
            )

        aa_sequence_tokens = precursor_tokens[
            (precursor_tokens.index(smiles_aa_sequence_separator) + 1) :
        ]
        predicted_site_indices = []

        cumulative_index = 0
        for i, token in enumerate(aa_sequence_tokens):
            if (i + 1) in filtered_sites:
                predicted_site_indices.append(
                    [cumulative_index, cumulative_index + len(token)]
                )
            cumulative_index += len(token)

        return predicted_site_indices

    def get_prediction_confidence_score(
        self,
        mapped_rxn: Any,
        ground_truth_aa_sequence_token_indices: List[Tuple[int, int]],
        rxn_separator: str = REACTION_SEPARATOR,
        smiles_aa_sequence_separator: str = SMILES_AA_SEQUENCE_SEPARATOR,
        aa_sequence: Optional[str] = None,
    ) -> Dict[str, Any]:
        """get the prediction confidence score from the mapped reaction and the ground truth amino acid residues
        Args:
            mapped_rxn: the mapped reaction
            ground_truth_enzyme_token: list of the boundaries of the amino acid residues experimentally identified as active sites
            rxn_separator: token used to separate reactants and enzyme from products
            smiles_aa_sequence_separator: token used to separate smiles enzyme
            site_threshold_frequency: the minimum frequency for an aa_sequence_token to be considered as active site
            aa_sequence: amino acid sequence

        Returns:
            a dictionary containing the overlap score the penalty score and the false positive rate
        """
        predicted_site_indices = self.get_predicted_active_site(
            mapped_rxn,
            rxn_separator=rxn_separator,
            smiles_aa_sequence_separator=smiles_aa_sequence_separator,
        )

        return self.get_overlap_and_penalty_score(
            predicted_site_indices,
            ground_truth_aa_sequence_token_indices,
            aa_sequence=aa_sequence,
        )

    def get_interaction_dictionary(
        self, mol: str, pattern: str = INTERACTION_PATTERN
    ) -> Tuple[str, Dict[str, int]]:
        """Extract the interaction dictionary.

        Args:
            mol: string representation of the molecule from where interaction dictionary will be extracted.
            pattern: pattern used to indicate the interactions of the atoms|aa_residues of smiles|aa_sequence

        Returns:
            a tuple containing a molecule (as string) and a dictionary of interaction.
                The key is the interaction number and the value if the index of the of
                the atoms|aa_residue where the interaction occurs.
        """
        interaction_list = re.findall(pattern, mol)
        interaction_list = [
            (interaction, mol.find(interaction)) for interaction in interaction_list
        ]
        interaction_list = sorted(interaction_list, key=lambda item: item[1])
        interaction_index_dic = {}

        while interaction_list:
            interaction, _ = interaction_list.pop(0)
            index = mol.find(interaction)
            interaction_index_dic[interaction] = index - 1
            mol = mol.replace(interaction, "", 1)

        return mol, interaction_index_dic

    def parse_rxn(
        self,
        smiles_aa_sequence: str,
        smiles_aa_sequence_separator: str = SMILES_AA_SEQUENCE_SEPARATOR,
    ) -> Dict[str, Any]:
        """Parse the reaction and retrieve insightful information.

        Args:
            smiles_aa_sequence: precursors of the reactions
            smiles_aa_sequence_separator: token used to separated the smiles from the amino acid sequence

        Returns:
            dictionary:
                - active_site: list of the active sites
            if `detailed_output=True`:
                - smiles: filtered smiles representation of the reactant
                - aa_sequence: filtered amino acid sequence of the enzyme.
        """
        pattern = re.compile(r"\s+")
        smiles_aa_sequence_tuned = re.sub(pattern, "", smiles_aa_sequence)
        for key, value in AA_DICT.items():
            smiles_aa_sequence_tuned = smiles_aa_sequence_tuned.replace(key, value)

        smiles, aa_sequence = smiles_aa_sequence_tuned.split(
            smiles_aa_sequence_separator
        )
        smiles, interaction_index_dic_smiles = self.get_interaction_dictionary(smiles)
        (
            aa_sequence,
            interaction_index_dic_aa_sequence,
        ) = self.get_interaction_dictionary(aa_sequence)

        smiles_index_map = {}
        for interaction, index in interaction_index_dic_smiles.items():
            if index not in smiles_index_map:
                smiles_index_map[index] = {
                    interaction_index_dic_aa_sequence[interaction]
                }
            else:
                smiles_index_map[index].add(
                    interaction_index_dic_aa_sequence[interaction]
                )

        active_sites_set = set()
        for interaction, index in interaction_index_dic_aa_sequence.items():
            active_sites_set.add((index, index + 1))
        active_sites_list = list(active_sites_set)

        output = {
            "active_site": active_sites_list,
            "smiles": smiles,
            "aa_sequence": aa_sequence,
        }
        return output

    def get_score_rxn(
        self,
        rxns: List[str],
        smiles_aa_sequence_separator: str = SMILES_AA_SEQUENCE_SEPARATOR,
        detailed_output: bool = False,
    ) -> List[Dict[str, Any]]:
        """score the left hand side of an enzymatic reaction
        Args:
            rxn: the enzymatic reaction
            smiles_aa_sequence_separator: token used to separate smiles from amino acid sequence
            detailed_output: flag to add whether or not additional information in the result
        """
        parsed_rxns = [self.parse_rxn(rxn) for rxn in rxns]
        filtered_rxns = [
            smiles_aa_sequence_separator.join([rxn["smiles"], rxn["aa_sequence"]])
            for rxn in parsed_rxns
        ]
        mapped_rxns = self.get_reactant_aa_sequence_attention_guided_maps(
            filtered_rxns, detailed_output=detailed_output
        )
        output = []
        for mapped_rxn, parsed_rxn in zip(mapped_rxns, parsed_rxns):
            result = self.get_prediction_confidence_score(
                mapped_rxn["mapped_rxn"],
                parsed_rxn["active_site"],
                aa_sequence=parsed_rxn["aa_sequence"],
            )
            result["mapped_rxn"] = mapped_rxn["mapped_rxn"]
            if detailed_output:
                result["confidence"] = mapped_rxn["confidence"]
                result["rxp_mapping_vector"] = mapped_rxn["rxp_mapping_vector"]
                result["rxp_confidences"] = mapped_rxn["rxp_confidences"]
                result["mapping_tuples"] = mapped_rxn["mapping_tuples"]
                result["rxp_mapping_vector"] = mapped_rxn["rxp_mapping_vector"]
                result["tokensxtokens_atten"] = mapped_rxn["tokensxtokens_atten"]
            output.append(result)
        return output

    def get_active_site_indices(
        self, aa_sequence: str, list_active_sites: List[Tuple[int, int]]
    ) -> List[int]:
        """Get the list of tokens which represent the active site of the given enzyme.

        Args:
            aa_sequence: amino acid sequence
            list_active_sites: list of the boundaries of the sub-sequences identified as the active site

        Returns:
            list of the token indices that wrap the active site.
        """
        aa_sequence_tokens = self.tokenizer.tokenizer.aa_sequence_tokenizer(aa_sequence)
        results = []
        for start_index, end_index in list_active_sites:
            cum_index = 0
            for index, token in enumerate(aa_sequence_tokens):
                if max(start_index, cum_index) <= min(
                    end_index, cum_index + len(token)
                ):
                    results.append(index)
                cum_index += len(token)

        return results

    def get_attention_active_site(
        self, rxns: List[str], list_of_active_site_lists: List[List[Tuple[int, int]]]
    ) -> List[Dict[str, Union[str, np.ndarray]]]:
        """Get the attention score of the reactant-protein interaction.

        Args:
            rxn: list of reactions.
            list_of_active_site_lists: list of list of the boundaries of the sub-sequences identified as the active site.

        Returns:
            tuple made up of the amino acid sequence, the attention scores of the active site tokens and,
                the uniform distribution of the attention weight.
        """
        if len(rxns) != len(list_of_active_site_lists):
            self.logger.error(
                "the list of the reaction should have the same length as the list of the list of the active sites"
            )
            raise ValueError(
                "the list of the reaction should have the same length as the list of the list of the active sites"
            )

        list_aa_sequences = []
        for rxn in rxns:
            aa_sequence = ""
            try:
                precursors, _ = rxn.split(REACTION_SEPARATOR)
            except Exception:
                precursors = rxn
            try:
                _, aa_sequence = precursors.split(SMILES_AA_SEQUENCE_SEPARATOR)
                list_aa_sequences.append(aa_sequence)
            except Exception:
                self.logger.error(
                    f"amino acid sequence not found in the reaction : {rxn}"
                )
                raise ValueError(
                    f"amino acid sequence not found in the reaction : {rxn}"
                )

        output_mapped = self.get_reactant_aa_sequence_attention_guided_maps(
            rxns, detailed_output=True
        )
        results = []
        for aa_sequence, list_active_site, out in zip(
            list_aa_sequences, list_of_active_site_lists, output_mapped
        ):
            attn = out["rxppxr_attns"]
            active_site_indices = np.array(
                self.get_active_site_indices(aa_sequence, list_active_site)
            )
            uniform_distribution = np.repeat(
                1 / attn.shape[1], active_site_indices.shape[0]
            )
            attention_active_site = np.max(attn[:, active_site_indices], axis=0)
            results.append(
                {
                    "aa_sequence": aa_sequence,
                    "active_site_indices": active_site_indices,
                    "attention_active_site": attention_active_site,
                    "uniform_distribution": uniform_distribution,
                }
            )

        return results

    def get_residue_attentions(self, rxns: List[str]):
        """Get the max attention value for each residue of the enzyme.

        Args:
            rxn: list of reactions

        Returns:
            a list of tuple made up of the aa_sequence of the list of the attention values.
        """
        rxns = [process_reaction(rxn) for rxn in rxns]
        attns = self.get_reactant_left_hand_side_attention(rxns)
        output = []
        for attn, rxn in zip(attns, rxns):
            precursors, _ = rxn.split(REACTION_SEPARATOR)
            _, aa_sequence = precursors.split(SMILES_AA_SEQUENCE_SEPARATOR)
            just_tokens = [
                token.replace("_", "") for token in self.tokenizer.tokenize(precursors)
            ]
            aa_sequence_tokens = just_tokens[
                (just_tokens.index(SMILES_AA_SEQUENCE_SEPARATOR) + 1) :
            ]
            attn_matrix = attn.cpu().numpy()
            attn_scorer = AttentionScorer(
                rxn,
                just_tokens,
                attn_matrix,
                attention_multiplier=float(self.attention_multiplier),
                top_k=int(self.top_k),
            )
            residue_attns = []
            aa_sequence_token_attns = attn_scorer.rxp_filt_atoms.mean(axis=0).tolist()
            for token, attn_value in zip(aa_sequence_tokens, aa_sequence_token_attns):
                residue_attns.extend([attn_value] * len(token))

            output.append((aa_sequence, residue_attns))
        return output

    def attach_attention_to_pdb(
        self,
        rxn: str,
        pdb_filename,
        value_transform_fn: Callable[[float], float] = lambda a: a,
    ):
        """Attach attention values to the amino acid sequence involved in the reaction
        Args:
            rxn: reaction
            pdb_filename: filepath of the pdb structure of the catalyst of the reaction
            value_transform: function to scale up/dowm the attention value for the plot.
        """
        attn_values = self.get_residue_attentions([rxn])[0][1]
        parser = PDBParser()
        structure = parser.get_structure("", pdb_filename)
        residues = list(structure.get_residues())

        for residue, attn_value in zip(residues, attn_values):
            for atom in residue.get_atoms():
                atom.set_bfactor(value_transform_fn(attn_value))

        io = PDBIO()
        io.set_structure(structure)
        io.save(pdb_filename)

    def binary_active_site_coloration(
        self,
        rxn: str,
        pdb_input_file: str,
        pdb_output_file: str,
        bfactor_value: int = 1,
    ):
        """Binary coloration of the three-dimensional structure of the enzyme.
        Args:
            rxn: the reaction
            pdb_input_file: filepath of the pdb structure of the catalyst of the reaction
            pdb_output_file: output file of the binary colored structure
            bfactor: temperature factore
        """

        out_mapper = self.get_reactant_aa_sequence_attention_guided_maps([rxn])[0]
        predicted_active_sites = self.get_predicted_active_site(
            out_mapper["mapped_rxn"]
        )
        precursors, _ = rxn.split(REACTION_SEPARATOR)
        _, aa_sequence = precursors.split(SMILES_AA_SEQUENCE_SEPARATOR)
        active_site_indices = set(
            self.get_active_site_indices(aa_sequence, predicted_active_sites)
        )
        aa_sequence_tokens = self.tokenizer.tokenizer.aa_sequence_tokenizer(aa_sequence)
        active_residue_indices: Set[int] = set()

        index = 0
        for i, token in enumerate(aa_sequence_tokens):
            if i in active_site_indices:
                active_residue_indices.update(range(index, index + len(token)))
            index += len(token)

        parser = PDBParser()
        structure = parser.get_structure("", pdb_input_file)
        residues = structure.get_residues()

        for i, res in enumerate(residues):
            bfactor = bfactor_value if i in active_residue_indices else 0
            for atom in res.get_atoms():
                atom.set_bfactor(bfactor)

        io = PDBIO()
        io.set_structure(structure)
        io.save(pdb_output_file)
