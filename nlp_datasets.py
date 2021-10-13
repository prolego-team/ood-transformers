"""
utilities for interacting with publically available datasets
"""

from typing import List

from nltk.corpus import reuters


def clean_text_string(text: str) -> str:
    """
    Remove line endings and extra spaces between words.
    """
    return " ".join([t.strip() for t in text.split()])


# Reuters
def reuters_dataset_dictionaries() -> List[dict]:
    """
    Convert the reuters multilabel dataset from nltk to a
    list of dictionaries with keys "text" and "labels".
    Also include a key "is_train", which is True for training examples
    and False for test examples.
    Assumes the reuters dataset has already been downloaded (see README.md).
    """

    document_ids = reuters.fileids()
    out = []
    for document_id in document_ids:
        # get the text (with line endings removed)
        text = clean_text_string(reuters.raw(document_id))
        labels = reuters.categories(document_id)
        is_train = document_id.startswith("training")
        out.append({"guid": document_id,
                    "text": text,
                    "labels": labels,
                    "is_train": is_train})
    return out


def reuters_class_labels() -> List[str]:
    return reuters.categories()
