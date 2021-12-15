"""Utility function to proprocess reaction"""
from functools import partial
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO
from rdkit import Chem, rdBase

# rdBase.DisableLog("rdApp.error")
rdBase.DisableLog("rdApp.warning")

BAD_TOKS = ["[CLS]", "[SEP]"]  # Default Bad Tokens
REACTION_SEPARATOR = ">>"
SMILES_AA_SEQUENCE_SEPARATOR = "|"
AA_DICT = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
}
INTERACTION_PATTERN = r"{\d*}"
PRIME = 521


class NotCanonicalizableSmilesException(ValueError):
    pass


class NotReactionException(ValueError):
    pass


def canonicalize_smi(smi: str, remove_atom_mapping=False) -> str:
    """Convert a SMILES string into its canonicalized form
    Args:
        smi: Reaction SMILES
        remove_atom_mapping: If True, remove atom mapping information from the canonicalized SMILES output

    Returns:
        SMILES reaction, canonicalized, as a string
    """
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        raise NotCanonicalizableSmilesException("Molecule not canonicalizable")
    if remove_atom_mapping:
        for atom in mol.GetAtoms():
            if atom.HasProp("molAtomMapNumber"):
                atom.ClearProp("molAtomMapNumber")
    return Chem.MolToSmiles(mol)


def process_reaction(
    rxn: str,
    fragments: str = "",
    fragment_bond: str = "~",
    reaction_separator: str = ">>",
    smiles_aa_sequence_separator: str = "|",
) -> str:
    """
    Remove atom-mapping, remove enzyme to reactants and canonicalize reaction.
    If fragment group information is given, keep the groups together using
    the character defined with fragment_bond.
    Args:
        rxn: Reaction SMILES
        fragments: (optional) fragments information
        fragment_bond: fragment bond
        reaction_separator: token used to separate the reactants and enzyme to the products
        smiles_aa_sequence_separator: token used to separate the reactants to the enzyme

    Returns:
        joined_precursors>>joined_products reaction SMILES_aa_sequence
    """

    # split the reactions into reactants, enzyme and products
    reactants, aa_sequence, products = "", "", ""
    try:
        precursors, products = rxn.split(reaction_separator)
    except Exception:
        precursors = rxn
    try:
        reactants, aa_sequence = precursors.split(smiles_aa_sequence_separator)
    except Exception:
        reactants = precursors

    # canonicalize the smiles component of the reaction (reactants and products)
    try:
        reactants_list = [canonicalize_smi(r, True) for r in reactants.split(".")]
        products_list = [canonicalize_smi(p, True) for p in products.split(".")]
    except NotCanonicalizableSmilesException:
        return ""

    if len(fragments) > 1 and fragments[1] == "f":
        number_of_reactants = len(reactants_list)
        groups = fragments[3:-1].split(",")
        new_reactants = reactants_list.copy()
        new_products = products_list.copy()

        for group in groups:
            grouped_smi = []
            if group.startswith("ˆ"):
                return ""
            for member in group.split("."):
                member_number = int(member)
                if member_number >= number_of_reactants:
                    grouped_smi.append(
                        products_list[member_number - number_of_reactants]
                    )
                    products_list.remove(
                        products_list[member_number - number_of_reactants]
                    )
                else:
                    grouped_smi.append(products_list[member_number])
                    new_reactants.remove(products_list[member_number])
            if member_number >= number_of_reactants:
                new_products.append(fragment_bond.join(sorted(grouped_smi)))
            else:
                new_reactants.append(fragment_bond.join(sorted(grouped_smi)))
        reactants_list, products_list = new_reactants, new_products
    # join the list of reactants and the list of products
    reactants, products = ".".join(reactants_list), ".".join(products_list)

    output = f"{reactants}{smiles_aa_sequence_separator}{aa_sequence}{reaction_separator}{products}"
    return output


def is_atom(token: str, special_tokens: List[str] = BAD_TOKS) -> bool:
    """Determine whether a token is an atom.
    Args:
        token: Token fed into the transformer model
        special_tokens: List of tokens to consider as non-atoms (often introduced by tokenizer)

    Returns:
        bool: True if atom, False if not
    """
    bad_toks = set(special_tokens)
    normal_atom = token[0].isalpha() or token[0] == "["
    is_bad = token in bad_toks
    return (not is_bad) and normal_atom


def get_mask_for_tokens(tokens: List[str], special_tokens: List[str] = []) -> List[int]:
    """Return a mask for a tokenized smiles, where atom tokens
    are converted to 1 and other tokens to 0.

    e.g. c1ccncc1 would give [1, 0, 1, 1, 1, 1, 1, 0]

    Args:
        smiles: Smiles string of reaction
        special_tokens: Any special tokens to explicitly not call an atom. E.g. "[CLS]" or "[SEP]"

    Returns:
        Binary mask as a list where non-zero elements represent atoms
    """
    check_atom = partial(is_atom, special_tokens=special_tokens)
    atom_token_mask = [1 if check_atom(t) else 0 for t in tokens]

    return atom_token_mask


def number_tokens(tokens: List[str], special_tokens: List[str] = BAD_TOKS) -> List[int]:
    """Map list of tokens to a list of numbered atoms
    Args:
        tokens: Tokenized SMILES
        special_tokens: List of tokens to not consider as atoms
    Example:
        >>> number_tokens(['[CLS]', 'C', '.', 'C', 'C', 'C', 'C', 'C', 'C','[SEP]'])
                #=> [-1, 0, -1, 1, 2, 3, 4, 5, 6, -1]
    """
    atom_num = 0
    isatm = partial(is_atom, special_tokens=special_tokens)

    def check_atom(t):
        if isatm(t):
            nonlocal atom_num
            ind = atom_num
            atom_num = atom_num + 1
            return ind
        return -1

    out = [check_atom(t) for t in tokens]

    return out


def is_mol_end(a: str, b: str) -> bool:
    """Determine if `a` and `b` are both tokens within a molecule (Used by the `group_with` function).

    Returns False whenever either `a` or `b` is a molecule delimeter (`.` or `>>`)"""
    no_dot = (a != ".") and (b != ".")
    no_arrow = (a != ">>") and (b != ">>")
    no_pipe = (a != "|") and (b != "|")

    return no_dot and no_arrow and no_pipe


def group_with(predicate, xs: List[Any]):
    """Takes a list and returns a list of lists where each sublist's elements are
    all satisfied pairwise comparison according to the provided function.
    Only adjacent elements are passed to the comparison function
        Original implementation here: https://github.com/slavaGanzin/ramda.py/blob/master/ramda/group_with.py
        Args:
            predicate ( f(a,b) => bool): A function that takes two subsequent inputs and returns True or Fale
            xs: List to group
    """
    out = []
    is_str = isinstance(xs, str)
    group = [xs[0]]

    for x in xs[1:]:
        if predicate(group[-1], x):
            group += [x]
        else:
            out.append("".join(group) if is_str else group)
            group = [x]

    out.append("".join(group) if is_str else group)

    return out


def split_into_mols(tokens: List[str]) -> List[List[str]]:
    """Split a reaction SMILES into SMILES for each molecule"""
    split_toks = group_with(is_mol_end, tokens)
    return split_toks


def tokens_to_smiles(tokens: List[str], special_tokens: List[str] = BAD_TOKS) -> str:
    """Combine tokens into valid SMILES string, filtering out special tokens
    Args:
        tokens: Tokenized SMILES
        special_tokens: Tokens to not count as atoms

    Returns:
        SMILES representation of provided tokens, without the special tokens
    """
    bad_toks = set(special_tokens)
    return "".join([t for t in tokens if t not in bad_toks])


def get_adjacency_matrix(smiles: str):
    """
    Compute adjacency matrix between atoms. Only works for single molecules atm and not for rxns
    Args:
        smiles: SMILES representation of a molecule

    Returns:
        Numpy array representing the adjacency between each atom and every other atom in the molecular SMILES.
        Equivalent to `distance_matrix[distance_matrix == 1]`
    """

    mol = Chem.MolFromSmiles(smiles)
    return Chem.GetAdjacencyMatrix(mol)


def tokens_to_adjacency(tokens: List[str]) -> np.ndarray:
    """Convert a tokenized reaction SMILES into a giant adjacency matrix.
    Note that this is a large, sparse Block Diagonal matrix of the adjacency matrix for each molecule in the reaction.
    Args:
        tokens: Tokenized SMILES representation

    Returns:
        Numpy Array, where non-zero entries in row `i` indicate the tokens that are atom-adjacent to token `i`
    """
    from scipy.linalg import block_diag

    mol_tokens = split_into_mols(tokens)
    separator_set = {".", ">>", "|"}
    # select the atom that doesn't appear in the separator set of the tokens
    # altered_mol_tokens = [m for m in mol_tokens if separator_set.isdisjoint(set(m))]

    smiles = [tokens_to_smiles(mol, BAD_TOKS) for mol in mol_tokens]
    altered_smiles = [s for s in smiles if s not in separator_set]
    adjacency_mats = [get_adjacency_matrix(s) for s in altered_smiles]
    rxn_mask = block_diag(*adjacency_mats)

    return rxn_mask


def get_atom_types_smiles(smiles: str) -> List[int]:
    """Convert each atom in a SMILES into a list of their atomic numbers
    Args:
        smiles: SMILES representation of molecule

    Returns:
        List of atom numbers for each atom in the smiles. Reports atoms in the same order they were passed in the original SMILES
    """
    smiles_mol = Chem.MolFromSmiles(smiles)
    atom_types = [atom.GetAtomicNum() for atom in smiles_mol.GetAtoms()]
    return atom_types


def generate_atom_mapped_reaction_atoms(
    rxn: str, reactant_atom_map: Dict[int, List[int]], expected_atom_maps=None
):
    """Generate atom-mapped reaction from unmapped reaction and
    AA-2-reactant atoms mapping vector.

    Args:
        rxn: unmapped reaction
        reactant_atom_map: reactant to aa_sequence mapping
        expected_atom_maps: (optional) if given return the differences

    Returns:
        results for the atom-mapped reaction.
    """
    precursors, products = rxn.split(REACTION_SEPARATOR)
    reactants, aa_sequence = precursors.split(SMILES_AA_SEQUENCE_SEPARATOR)
    reactants_mol = Chem.MolFromSmiles(reactants)

    reactant_atom_maps = []
    differing_maps = []

    rxn_mapped: Union[str, Dict[str, Any]]

    for i, atom in enumerate(reactants_mol.GetAtoms()):
        if not reactant_atom_map[i] == -1:
            corresponding_aa_sequence_token_map = encode_aa_sequence_indices(
                [j + 1 for j in reactant_atom_map[i]]
            )
            reactant_atom_maps.append(corresponding_aa_sequence_token_map)
            atom.SetProp("molAtomMapNumber", corresponding_aa_sequence_token_map)

            if expected_atom_maps is not None and (
                expected_atom_maps[i] not in reactant_atom_map[i]
            ):
                differing_maps.append(corresponding_aa_sequence_token_map)

    try:
        reactant_mapped = Chem.MolToSmiles(reactants_mol)
        rxn_mapped = f"{reactant_mapped}{SMILES_AA_SEQUENCE_SEPARATOR}{aa_sequence}{REACTION_SEPARATOR}{products}"
    except Exception:
        rxn_mapped = {"rxn": rxn, "reactants_mol": reactants_mol}

    output: Any = (rxn_mapped,)
    if expected_atom_maps is not None:
        output += (differing_maps,)

    return output


def encode_aa_sequence_indices(indices: List[int]) -> str:
    """Encode the list of aa_sequence token indices in a unique number in the basis `prime`
    Args:
        indices: list of aa_sequence tokens

    Returns:
        unique number cast into string
    """
    output, factor = 0, 1
    for index in indices:
        output += index * factor
        factor *= PRIME
    return str(output)


def decode_aa_sequence_indices(num: int) -> List[int]:
    """Number decomposition
    Args:
        num: number to break down

    Returns:
        list of the cordinates of `num` the basis `prime`
    """
    output = []
    while num:
        output.append(num % PRIME)
        num = num // PRIME
    return output


def levenshtein(seq1: str, seq2: str) -> int:
    """Levenshtein distance between two sequences
    Args:
        seq1: first sequence
        seq2: second sequence

    Returns:
        the levenshtein distance between the two sequences fed as arguments
    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1, matrix[x - 1, y - 1], matrix[x, y - 1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1, matrix[x - 1, y - 1] + 1, matrix[x, y - 1] + 1
                )
    return matrix[size_x - 1, size_y - 1]


def pairwise_levenshtein(list_seq1: List[str], list_seq2: List[str]) -> float:
    """Minimum levenshtein distance between two lists of sequence
    Args:
        list_seq1: first list of sequence
        list_seq2: second list of sequence

    Returns:
        minimum pairwise levenshtein distance between the sequences of the two lists.
    """
    scores: List[float] = []
    if len(list_seq1) == 0:
        scores = [len(seq2) for seq2 in list_seq2]
        return np.mean(scores)
    if len(list_seq2) == 0:
        scores = [len(seq1) for seq1 in list_seq1]
        return np.mean(scores)
    for seq2 in list_seq2:
        score = 1e5
        for seq1 in list_seq1:
            score = min(score, levenshtein(seq1, seq2))
        scores.append(score)
    return np.mean(scores)


def clean_pdb_bfactor(input_pdb: str) -> None:
    """Set the temperature factor to 0
    Args:
        input_pdb: filepath of the pdb-structure
    """
    parser = PDBParser()
    structure = parser.get_structure("id", input_pdb)

    for model in structure:
        for _, chain in sorted(model.child_dict.items()):
            for res in chain.get_unpacked_list():
                for atom in res.get_atoms():
                    atom.set_bfactor(0.0)

    io = PDBIO()
    io.set_structure(structure)
    io.save(input_pdb)


def annotate_pdb_structure(
    aa_sequence: str,
    list_active_site: List[Tuple[int, int]],
    input_pdb: str,
    output_pdb: str,
) -> None:
    """Color a pdb structure according to the list of active site given as parameter
    Args:
        aa_sequence: amino acid sequence.
        list_active_site: list of active portion that should be vibrant.
        input_pdb: input filepath of the structure to color
        output_pdb: output filepath where the colored structure will be save.
    """
    list_active_site = sorted(list_active_site, key=lambda item: item[0])
    active_residue_indices = set()
    i, up_bound = 0, len(aa_sequence)
    while i < up_bound:
        if len(list_active_site) == 0:
            break
        site = list_active_site[0]
        if i >= site[0]:
            if i >= site[1]:
                list_active_site.pop(0)
                continue
            else:
                active_residue_indices.add(i)
        i += 1

    parser = PDBParser()
    structure = parser.get_structure("", input_pdb)

    for i, res in enumerate(structure.get_residues()):
        bfactor = 1 if i in active_residue_indices else 0
        for atom in res.get_atoms():
            atom.set_bfactor(bfactor)

    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)


def intersect_pdb_annotation_with_model_mapping(
    input_pdb_1: str, input_pdb_2: str, chain_id: str, output_pdb: str
) -> None:
    """draw the intersection of two pdb structure coloration
    Args:
        input_pdb_1: filepath of the first configuration
        input_pdb_2: filepath of the second configuration
        chain_id: id of the amino acid chain to consider for the intersection
        output: filepath to save the merged pdb structure
    """
    parser = PDBParser()
    structure_1 = parser.get_structure("id1", input_pdb_1)
    chain_1 = structure_1[0].child_dict[chain_id]
    chain_2 = parser.get_structure("id2", input_pdb_2)[0].child_dict[chain_id]

    residues_1 = list(chain_1.get_unpacked_list())
    residues_2 = list(chain_2.get_unpacked_list())
    if len(residues_1) != len(residues_2):
        raise ValueError(
            f"structure 1 and 2 should have the same number of residues. {len(residues_1)} # {len(residues_2)}"
        )
    for res_1, res_2 in zip(residues_1, residues_2):
        for atom_1, atom_2 in zip(res_1.get_atoms(), res_2.get_atoms()):
            bf1, bf2 = atom_1.get_bfactor(), atom_2.get_bfactor()
            bf = max(bf1, bf2)
            if int(bf1) and int(bf2):
                bf = (bf1 + bf2) / 2
            atom_1.set_bfactor(bf)

    io = PDBIO()
    io.set_structure(structure_1)
    io.save(output_pdb)
