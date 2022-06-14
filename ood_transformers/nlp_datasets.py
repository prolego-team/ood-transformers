"""
utilities for interacting with publically available datasets
"""

from typing import List, Optional, Tuple
import random

from nltk.corpus import reuters, movie_reviews

from text_classification.dataset_utils import (
    InputMultilabelExample,
    dictionaries_to_input_multilabel_examples
)


def clean_text_string(text: str) -> str:
    """
    Remove line endings and extra spaces between words.
    """
    return " ".join([t.strip() for t in text.split()])


# Reuters
# the top five categories all have >500 examples distributed between train and test
TOP_FIVE_CATEGORIES = ["earn", "acq", "money-fx", "grain", "crude"]
BACKGROUND_CATEGORIES = ["trade", "interest", "ship", "wheat", "corn"]

def reuters_dataset_dictionaries(categories: Optional[List[str]]) -> List[dict]:
    """
    Convert the reuters multilabel dataset from nltk to a list of dictionaries
    with keys "text" and "labels".
    If categories are provided, only return the reuters data belonging to those
    categories, and modify labels to contain only those categories.
    Also include a key "is_train", which is True for training examples
    and False for test examples.
    Assumes the reuters dataset has already been downloaded (see README.md).
    """

    document_ids = reuters.fileids(categories)
    out = []
    for document_id in document_ids:
        # get the text (with line endings removed)
        text = clean_text_string(reuters.raw(document_id))
        labels = reuters.categories(document_id)
        if categories:
            # only store the categories of interest
            labels = [l for l in labels if l in categories]
        is_train = document_id.startswith("training")
        out.append({"guid": document_id,
                    "text": text,
                    "labels": labels,
                    "is_train": is_train})
    return out


def reuters_class_labels() -> List[str]:
    return reuters.categories()


def reuters_dataset_to_train_test_examples(
        categories: Optional[List[str]],
        background_categories: Optional[List[str]] = None,
        shuffle_train_examples: bool = False,
        seed: int = 12345
    ) -> Tuple[List[InputMultilabelExample], List[InputMultilabelExample]]:
    """
    Create lists of train and test examples for the Reuters dataset.
    If categories are provided, return only the examples that have at least one
       label that matches a category, and modify the list of labels to include
       only those categories. Otherwise, if categories is None, return all examples in
       the Reuters dataset, with no modifications to labels.
    If background_categories is not None, also include any examples that
       contain these background categories as labels, but remove the background
       labels.
    If shuffle_train_examples is true, shuffle the order of training examples (set seed
       using the seed argument)
    """
    # build list of all categories, including foreground and background
    all_categories = []
    if categories is None:
        # use ALL categories (by setting all_categories to None)
        all_categories = None
    else:
        all_categories += categories
        if background_categories:
            all_categories += background_categories

    reuters_data = reuters_dataset_dictionaries(categories=all_categories)
    train_examples = dictionaries_to_input_multilabel_examples(
        [d for d in reuters_data if d["is_train"]],
        False
    )
    test_examples = dictionaries_to_input_multilabel_examples(
        [d for d in reuters_data if not d["is_train"]],
        False
    )

    # remove labels for background categories
    if background_categories:
        def remove_background_labels(examples):
            new_examples = [InputMultilabelExample(
                example.guid,
                example.text,
                [l for l in example.labels if l not in background_categories])
                for example in examples]
            return new_examples
        train_examples = remove_background_labels(train_examples)
        test_examples = remove_background_labels(test_examples)

    if shuffle_train_examples:
        random.seed(seed)
        random.shuffle(train_examples)

    return train_examples, test_examples


def movie_reviews_dataset_dictionaries(categories: Optional[List[str]]) -> List[dict]:
    """
    Convert the movie reviews dataset from nltk to a list of dictionaries
       with keys "text" and "labels".
    If categories are provided, modify labels to contain only those categories.
    Assumes the movie_reviews dataset has already been downloaded (see README.md).
    """

    document_ids = movie_reviews.fileids()
    out = []
    for document_id in document_ids:
        text = clean_text_string(movie_reviews.raw(document_id))
        labels = movie_reviews.categories(document_id)
        if categories:
            labels = [l for l in labels if l in categories]
        out.append({"guid": document_id,
                    "text": text,
                    "labels": labels})
    return out


def movie_reviews_dataset_to_examples(
        categories: Optional[List[str]],
    ) -> List[InputMultilabelExample]:
    """
    Create lists of examples for movie reviews dataset. Include all examples
       regardless, but if categories are provided, modify example labels to
       contain only those categories.
    """
    movie_reviews_data = movie_reviews_dataset_dictionaries(categories=categories)
    examples = dictionaries_to_input_multilabel_examples(
        movie_reviews_data,
        False
    )
    return examples
