"""Baseline score definition"""
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from rxn_aa_mapper.aa_mapper import RXNAAMapper


def get_average_significant_active_site_tokens(
    input_data_dir: str,
    output_path: str,
    vocabulary_file: str,
    aa_sequence_tokenizer_filepath: str,
    model_path: str,
    top_k: int,
    head: int,
    layers: str,
    batch_size: int = 4,
    min_num_rxns: int = 10,
    p_value_threshold: float = 0.05,
) -> None:
    """Compute the average token with siginificant p-value

    Args:
        input_data_dir: path of the directory where the annotated reactions are stored
        output_path: path of the csv file where the result will be stored
        vocabulary_file: path of the vocabulary file of the tokenizer
        aa_sequence_tokenizer_filepath: path of the file containing the amino acid residue tokens
        model_path: path of the directory where the checkpoint is stored
        top_k: number of amino acid tokens to bind with each reactant's atom
        head: head at which the attention scores will be extracted
        layers: list of layers at which the attention scores will be extracted
        batch_size: batch size
        min_num_rxns: minimum number of reactions to apply the statistic test.
    """
    layers = eval(str(layers))
    if not isinstance(layers, list):
        raise ValueError(
            "layers should be a stringified list of the indices of the layers to select"
        )
    config_mapper = {
        "vocabulary_file": vocabulary_file,
        "aa_sequence_tokenizer_filepath": aa_sequence_tokenizer_filepath,
        "model_path": model_path,
        "top_k": top_k,
        "head": head,
        "layers": layers,
    }
    mapper = RXNAAMapper(config=config_mapper)
    attention_active_site_scores = defaultdict(list)
    uniform_distributions = defaultdict(list)
    dic_active_site_index = {}

    for filename in tqdm(list(os.listdir(input_data_dir))):
        input_filepath = os.path.join(input_data_dir, filename)

        for chunk in tqdm(pd.read_csv(input_filepath, chunksize=batch_size)):
            rxns = chunk["rxn"].tolist()
            list_list_active_site = list(map(eval, chunk["active_site"]))
            try:
                outputs = mapper.get_attention_active_site(rxns, list_list_active_site)
                for out in outputs:
                    attention_active_site_scores[out["aa_sequence"]].append(
                        out["attention_active_site"].tolist()
                    )
                    uniform_distributions[out["aa_sequence"]].append(
                        out["uniform_distribution"].tolist()
                    )
                    dic_active_site_index[out["aa_sequence"]] = list(
                        map(str, out["active_site_indices"])
                    )
            except Exception:
                continue

    output = []
    for aa_sequence, list_active_site in dic_active_site_index.items():
        attention_scores = pd.DataFrame(
            data=attention_active_site_scores[aa_sequence], columns=list_active_site
        )
        uniform_scores = pd.DataFrame(
            data=uniform_distributions[aa_sequence], columns=list_active_site
        )
        if len(attention_scores) < min_num_rxns:
            continue

        p_values = np.ones(len(list_active_site))
        for i, index in enumerate(list_active_site):
            x = attention_scores[index]
            y = uniform_scores[index]
            _, p_value = ranksums(x, y)
            p_values[i] = p_value

        output_test = multipletests(p_values, method="fdr_bh")
        significant_token = float(output_test[0].mean())
        output.append((aa_sequence, json.dumps(p_values.tolist()), significant_token))

    output = pd.DataFrame(
        data=output, columns=["aa-sequence", "p-values", "significant-token"]
    )
    output.to_csv(output_path, index=False)
