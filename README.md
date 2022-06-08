# ood-transformers

## Objective

Build an NLP model that can:

- identify out-of-set examples
    - known unknowns
    - unknown unknowns
- know when it's made a mistake
- identify mis-labeled training data

## Getting Started

### Code Repository

Clone the git repo:

        git clone https://github.com/prolego-team/ood-transformers.git

Create a virtual environment and install ood-transformers using pip:

        cd ood-transformers
        python3 -m venv .venv
        source .venv/bin/activate
        cd ..
        pip install -e ood-transformers

The environment only needs to be set up once. After it has been created, it can be activated using the command:

        cd ood-transformers
        source .venv/bin/activate

### Data

We will use the Reuters ModApte dataset and the Movie Reviews dataset, both of which are available for download using nltk:

        python
        import nltk

        nltk.download("reuters")
        nltk.download("movie_reviews")

### Objectosphere Experiment

Objective: Train a transformer for multilabel classification that demonstrates uncertainty in the face of unfamiliar (out-of-distribution) data.

Inspired by: "Reducing Network Agnostophobia", Akshay Raj Dhamija, Manuel Günther, & Terrance E. Boult,
https://arxiv.org/abs/1811.04110

To recreate results, follow these steps:

1. Train the models:

        python -m experiments.objectosphere_out_of_set --do_train

    This will create 3 directories ("trained_base", "trained_base_w_background", "trained_objectosphere"), each containing trained model artifacts.

2. Evaluate model performance on in-set data:

        python -m experiments.objectosphere_out_of_set --do_eval

    Performance metrics are printed in the console.

3. Compute AUC on in-set vs. out-of-set model performance scores:

        python -m experiments.objectosphere_out_of_set --do_auc

    AUCs are printed in the console.
