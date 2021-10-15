"""
fixtures for unit tests
"""

from typing import List

import pytest

from text_classification.dataset_utils import InputMultilabelExample


@pytest.fixture
def num_labels() -> int:
    """number of class labels"""
    return 4


@pytest.fixture
def class_labels(num_labels: int) -> List[str]:
    """list of class labels (strings)"""
    return ["Label " + str(i) for i in range(num_labels)]


@pytest.fixture
def input_multilabel_examples(class_labels: List[str]) -> List[InputMultilabelExample]:
    """list of 10 input multi-label examples, each labeled with all class labels"""
    return [InputMultilabelExample(str(i), "Text " + str(i), class_labels)
            for i in range(10)]