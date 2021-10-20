from typing import Optional, List

import pytest
from text_classification.dataset_utils import InputMultilabelExample

import nlp_datasets


@pytest.mark.parametrize("categories", [None, nlp_datasets.TOP_FIVE_CATEGORIES])
def test_reuters_dataset_dictionaries(categories: Optional[List[str]]) -> None:
    """
    Test that reuters_dataset_dictionaries returns the expected output type.
    When categories are specified, check that the list of labels only contains
    the specified categories.
    """

    reuters_data = nlp_datasets.reuters_dataset_dictionaries(categories)

    assert type(reuters_data) == list
    assert type(reuters_data[0]) == dict
    for key in ["guid", "text", "labels", "is_train"]:
        assert key in reuters_data[0].keys()

    if categories is None:
        assert len(reuters_data) == 10788
    else:
        assert len(reuters_data) == 8147
        # check that only the top five categories are included in the labels
        for data in reuters_data:
            for label in data["labels"]:
                assert label in nlp_datasets.TOP_FIVE_CATEGORIES


@pytest.mark.parametrize("categories", [None, nlp_datasets.TOP_FIVE_CATEGORIES])
def test_movie_reviews_dataset_to_examples(categories: Optional[List[str]]) -> None:
    """
    test that examples are created with expected set of labels
    """
    examples = nlp_datasets.movie_reviews_dataset_to_examples(categories)
    assert len(examples) == 2000
    for example in examples:
        assert type(example) == InputMultilabelExample
        if categories:
            # examples shouldn't have any categories (since the reuters top five
            # categories aren't included in this dataset)
            assert len(example.labels) == 0
        else:
            assert len(example.labels) > 0
