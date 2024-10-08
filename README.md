# RXNAAMapper

RXNAAMapper is a tool designed to identify binding sites in protein sequences by leveraging language models trained on biochemical reactions. The tool can capture the signal characterizing amino acid (AA) binding sites using linguistic representations for proteins and their molecular substrates, performing unsupervised binding site prediction from protein sequences and reaction SMILES.

## setup
To set up the environment, use the following commands:
```console
conda env create -f conda.yml
conda activate rxn_aa_mapper
```

In the following we consider the [examples](./examples) provided to show how to use RXNAAMapper.

## generate a vocabulary to be used with the `EnzymaticReactionBertTokenizer`

Create a vocabulary compatible with the enzymatic reaction tokenizer:

```console
create-enzymatic-reaction-vocabulary ./examples/data-samples/biochemical ./examples/token_75K_min_600_max_750_500K.json /tmp/vocabulary.txt "*.csv"
```

## use the tokenizer

The example below shows how to use the `LMEnzymaticReactionTokenizer` with the vocabulary previously created and the tokenizer:

```python
from rxn_aa_mapper.tokenization import LMEnzymaticReactionTokenizer

tokenizer = LMEnzymaticReactionTokenizer(
    vocabulary_file="./examples/vocabulary_token_75K_min_600_max_750_500K.txt",
    aa_sequence_tokenizer_filepath="./examples/token_75K_min_600_max_750_500K.json",
    aa_sequence_tokenizer_type="generic"
)
tokenizer.tokenize("NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1.O=C([O-])CC(C(=O)[O-])C(O)C(=O)[O-]|AGGVKTVTLIPGDGIGPEISAAVMKIFDAAKAPIQANVRPCVSIEGYKFNEMYLDTVCLNIETACFATIKCSDFTEEICREVAENCKDIK>>O=C([O-])CCC(=O)C(=O)[O-]")
```

## train the model

We use WandB for logging, if you don't have a mode configured you can simply disable it by setting:

```console
export WANDB_MODE=offline
```

The [`mlm-trainer`](./bin/mlm-trainer) script can be used to train a model via MTL:

```console
mlm-trainer \
    ./examples/data-samples/biochemical \ # just a sample train folder
    ./examples/data-samples/biochemical \  # just a sample validation folder
    ./examples/vocabulary_token_75K_min_600_max_750_500K.txt \
    /tmp/mlm-trainer-log \
    ./examples/sample-config.json \ # for a more realistic config see ./examples/config.json
    "*.csv" \
    1 \
    ./examples/data-samples/organic \ # just a sample train folder
    ./examples/data-samples/organic \  # just a sample validation folder
    ./examples/token_75K_min_600_max_750_500K.json \
    "generic"
```

Checkpoints will be stored in the `/tmp/mlm-trainer-log` for later usage in identification of active sites.

These checkpoints can be converted into a HuggingFace model with:

```console
checkpoint-to-hf-model /path/to/model.ckpt /tmp/rxnaamapper-pretrained-model ./examples/vocabulary_token_75K_min_600_max_750_500K.txt ./examples/sample-config.json ./examples/token_75K_min_600_max_750_500K.json
```

## predict active site

Once trained, the RXNAAMapper model can predict reactant atoms and map them to AA sequence locations, indicating potential binding sites:

```python
from rxn_aa_mapper.aa_mapper import RXNAAMapper

config_mapper = {
    "vocabulary_file": "./examples/vocabulary_token_75K_min_600_max_750_500K.txt",
    "aa_sequence_tokenizer_filepath": "./examples/token_75K_min_600_max_750_500K.json",
    "aa_sequence_tokenizer_type": "generic",
    "model_path": "/tmp/rxnaamapper-pretrained-model",
    "head": 3,
    "layers": [11],
    "top_k": 1,
}
mapper = RXNAAMapper(config=config_mapper)
mapper.get_reactant_aa_sequence_attention_guided_maps(["NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1.O=C([O-])CC(C(=O)[O-])C(O)C(=O)[O-]|AGGVKTVTLIPGDGIGPEISAAVMKIFDAAKAPIQANVRPCVSIEGYKFNEMYLDTVCLNIETACFATIKCSDFTEEICREVAENCKDIK>>O=C([O-])CCC(=O)C(=O)[O-]"])
```
*NOTE:* The model path should contain both the model binary file and the config.json. These files are generated from the model trained and converted to a HuggingFace model using the script provided in the previous section.

## citation

```bib
@article{teukam2024language,
  title={Language models can identify enzymatic binding sites in protein sequences},
  author={Teukam, Yves Gaetan Nana and Dassi, Lo{\"\i}c Kwate and Manica, Matteo and Probst, Daniel and Schwaller, Philippe and Laino, Teodoro},
  journal={Computational and Structural Biotechnology Journal},
  volume={23},
  pages={1929--1937},
  year={2024},
  publisher={Elsevier}
}
```
