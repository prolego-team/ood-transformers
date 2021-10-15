"""
unit tests for openmax.py
"""

from typing import List

import pytest
import numpy as np

from text_classification.configs import read_config_for_inference
from text_classification.dataset_utils import InputMultilabelExample

import openmax


@pytest.mark.usefixtures("input_multilabel_examples")
@pytest.mark.usefixtures("num_labels")
def test_examples_to_mean_logit(
        input_multilabel_examples: List[InputMultilabelExample],
        num_labels: int) -> None:
    inference_config = read_config_for_inference("test_data/inference_config.json")
    mean_logit = openmax.examples_to_mean_logit(input_multilabel_examples, inference_config)
    assert type(mean_logit) == np.ndarray
    assert len(mean_logit) == num_labels

