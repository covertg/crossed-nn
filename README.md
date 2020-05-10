# crossing-nn

## *v2: work in progresse* ...

This repo contains the code, data, and results of a study done for LING 380 at Yale, *Neurals Network & Language* with Prof. Bob Frank.

* Can we probe a language model’s induction of syntax by characterizing its ability to make inferences?
* Specifically: can GPT-2 correctly infer subject-verb relations given various forms of **crossing dependencies**? Do its successes or failures arise in syntactically meaningful ways?

## Setup

This code depends on `pytorch`, `transformers`, `numpy`, and `pandas`. Precise version numbers for the libraries and their dependencies are found in `requirements.txt`:

```
pip install -r requirements.txt
```

Additionally, data analysis done with R depends on `tidyverse`.

## Directory structure

* `crossing/`
    - Python package for generating templated data, running models, and running experiments.
    - `gpt2.py`: Loads and provides an interface to pretrained GPT-2.
    - `data.py`: Processes template and language data to generate test sentences.
    - `experiment.py`: Runs an experiment---read sentences and calculate surprisal over the respective region.
* `data/`
    - Input/output and data analysis.
    - `items.tsv`: Initial language data in 5-tuples (A, B, Vtr, Vin1, Vin2).
    - `template$i.tsv`: Map from 5-tuple to premises and entailments.
    - `intermediate_sents$i.tsv`: Premise-entailment sentences before expanding {curly braces}.
    - `sents$i.tsv`: Expanded sentences.
    - `results_MODEL-$i`.tsv: Model outputs.
    - `exploration.rmd`: Data exploration and analysis.
* `data_gen.ipynb`: Generates expanded sentences from 5-tuples.
* `run_experiment.ipynb`: Runs all experiments and saves results.

## Experiments, Results

See the Python notebooks and written report in this root directory!