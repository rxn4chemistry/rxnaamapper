# rxn-aa-mapper

Reactions SMILES-AA sequence mapping

## setup

```console
conda env create -f conda.yml
conda activate rxn_aa_mapper
```

In the following we consider on [examples](./examples) provided to show how to use RXNAAMapper.

## generate a vocabulary to be used with the `EnzymaticReactionBertTokenizer`

Create a vocabulary compatible with the enzymatic reaction tokenizer:

```console
create-enzymatic-reaction-vocabulary ./examples/data-samples/biochemical ./examples/token_75K_min_600_max_750_500K.json /tmp/vocabulary.txt "*.csv"
```

## use the tokenizer

Using the examples vocabulary and AA tokenizer provided, we can observe the enzymatic reaction tokenizer in action:

```python
from rxn_aa_mapper.tokenization import EnzymaticReactionBertTokenizer

tokenizer = EnzymaticReactionBertTokenizer(
    vocabulary_file="./examples/vocabulary_token_75K_min_600_max_750_500K.txt",
    aa_sequence_tokenizer_filepath="./examples/token_75K_min_600_max_750_500K.json"
)
tokenizer.tokenize("NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1.O=C([O-])CC(C(=O)[O-])C(O)C(=O)[O-]|AGGVKTVTLIPGDGIGPEISAAVMKIFDAAKAPIQANVRPCVSIEGYKFNEMYLDTVCLNIETACFATIKCSDFTEEICREVAENCKDIK>>O=C([O-])CCC(=O)C(=O)[O-]")
```

## train the model

The [`mlm-trainer`](./bin/mlm-trainer) script can be used to train a model via MTL:

```console
mlm-trainer \
    ./examples/data-samples/biochemical ./examples/data-samples/biochemical \  # just a sample, simply split data in a train and a validation folder
    ./examples/vocabulary_token_75K_min_600_max_750_500K.txt /tmp/mlm-trainer-log \
    ./examples/sample-config.json "*.csv" 1 \  # for a more realistic config see ./examples/config.json
    ./examples/data-samples/organic ./examples/data-samples/organic \  # just a sample, simply split data in a train and a validation folder
    ./examples/token_75K_min_600_max_750_500K.json
```

Checkpoints will be stored in the `/tmp/mlm-trainer-log` for later usage in identification of active sites.

Those can be turned into an HuggingFace model by simply running:

```console
checkpoint-to-hf-model /path/to/model.ckpt /tmp/rxnaamapper-pretrained-model ./examples/vocabulary_token_75K_min_600_max_750_500K.txt ./examples/sample-config.json ./examples/token_75K_min_600_max_750_500K.json
```

## predict active site

The trained model can used to map reactant atoms to AA sequence locations that potentially represent the active site.

```python
from rxn_aa_mapper.aa_mapper import RXNAAMapper

config_mapper = {
    "vocabulary_file": "./examples/vocabulary_token_75K_min_600_max_750_500K.txt",
    "aa_sequence_tokenizer_filepath": "./examples/token_75K_min_600_max_750_500K.json",
    "model_path": "/tmp/rxnaamapper-pretrained-model",
    "head": 3,
    "layers": [11],
    "top_k": 1,
}
mapper = RXNAAMapper(config=config_mapper)
mapper.get_reactant_aa_sequence_attention_guided_maps(["NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1.O=C([O-])CC(C(=O)[O-])C(O)C(=O)[O-]|AGGVKTVTLIPGDGIGPEISAAVMKIFDAAKAPIQANVRPCVSIEGYKFNEMYLDTVCLNIETACFATIKCSDFTEEICREVAENCKDIK>>O=C([O-])CCC(=O)C(=O)[O-]"])
```

## citation

```bib
@article{dassi2021identification,
  title={Identification of Enzymatic Active Sites with Unsupervised Language Modeling},
  author={Dassi, Lo{\"\i}c Kwate and Manica, Matteo and Probst, Daniel and Schwaller, Philippe and Teukam, Yves Gaetan Nana and Laino, Teodoro},
  year={2021}
  conference={AI for Science: Mind the Gaps at NeurIPS 2021, ELLIS Machine Learning for Molecule Discovery Workshop 2021}
}
```
