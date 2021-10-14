import pytest

import nlp_datasets


@pytest.mark.parametrize("categories", [None, nlp_datasets.TOP_FIVE_CATEGORIES])
def test_reuters_dataset_dictionaries(categories):
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
