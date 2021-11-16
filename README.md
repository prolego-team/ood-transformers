# openset-nlp

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

        git clone https://github.com/prolego-team/openset-nlp.git

Create a virtual environment and install package dependencies using pip:

        cd openset-nlp
        python3 -m venv .venv
        source .venv/bin/activate
        pip install -r requirements.txt

The environment only needs to be set up once. After it has been created, it can be activated using the command:

        cd openset-nlp
        source .venv/bin/activate

Clone the textclassification repo within openset-nlp:

        cd openset-nlp
        git clone https://github.com/prolego-team/textclassification.git
        pip install -e textclassification

### Data

We will use the Reuters ModApte dataset and the Movie Reviews dataset, both of which are available for download using nltk:

        python
        import nltk

        nltk.download("reuters")
        nltk.download("movie_reviews")
