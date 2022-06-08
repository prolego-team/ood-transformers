"""
unit tests for eval.py
"""

from typing import List

import pytest

from text_classification.dataset_utils import InputMultilabelExample, OutputMultilabelExample
from ood_transformers import eval


@pytest.mark.usefixtures("input_multilabel_examples")
def test_incorrect_prediction_aucs(
    input_multilabel_examples: List[InputMultilabelExample]) -> None:
    """
    test that incorrect_prediction_aucs generates
    the expected AUC values for some dummy data
    """

    test_examples = input_multilabel_examples
    # generate some dummy data
    test_examples = [
        InputMultilabelExample(
            test_example.guid,
            test_example.text,
            ["Label 0", "Label 1"]
        )
        for test_example in test_examples
    ]
    prediction_examples = [
        OutputMultilabelExample(
            test_example.guid,
            test_example.text,
            ["Label 0", "Label 1", "Label 2", "Label 3"],
            [0.9, 0.1, 0.9, 0.1]
        )
        for test_example in test_examples
    ]

    positive_auc, negative_auc = eval.incorrect_prediction_aucs(
        test_examples,
        prediction_examples,
        lambda confidence: confidence >= 0.5,
        lambda confidence: confidence < 0.5)

    assert positive_auc == 0.5
    assert negative_auc == 0.5
