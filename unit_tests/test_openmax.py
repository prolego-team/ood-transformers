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
def test_OpenMaxPredictor(
        input_multilabel_examples: List[InputMultilabelExample]
    ) -> None:
    """
    test that OpenMaxPredictor predicts based on distance threshold
    """
    inference_config = read_config_for_inference("test_data/inference_config.json")
    mean_logits = {k: np.ones(inference_config.num_labels)
                   for k in inference_config.class_labels}
    predictor = openmax.OpenMaxPredictor(
        inference_config.model_config,
        inference_config.class_labels,
        mean_logits)

    # run predictor with large distance threshold (should results in all examples having
    # all labels)
    predicted_examples = predictor(
        input_multilabel_examples, inference_config.max_length, np.inf)
    for example in predicted_examples:
        assert len(example.labels) == 4

    # run predictor with small distance threshold (should result in all examples having
    # no labels)
    predicted_examples = predictor(
        input_multilabel_examples, inference_config.max_length, -1.0)
    for example in predicted_examples:
        assert len(example.labels) == 0


@pytest.mark.usefixtures("input_multilabel_examples")
@pytest.mark.usefixtures("num_labels")
def test_examples_to_mean_logit(
        input_multilabel_examples: List[InputMultilabelExample],
        num_labels: int) -> None:
    """
    check that mean logit is correctly computed from all logits
    """
    inference_config = read_config_for_inference("test_data/inference_config.json")
    mean_logit = openmax.examples_to_mean_logit(
        input_multilabel_examples, inference_config)
    assert type(mean_logit) == np.ndarray
    assert len(mean_logit) == num_labels


@pytest.mark.parametrize("distance_function",
                         [openmax.euclidean_distance_function,
                          openmax.mae_distance_function,
                          openmax.fractional_absolute_distance_function,
                          openmax.fractional_euclidean_distance_function,
                          openmax.non_member_class_distance,
                          openmax.member_class_distance])
def test_distance_functions(distance_function) -> None:
    """
    unit tests for distance functions
    """

    n_examples = 10
    n_classes = 5
    mean_logits = np.random.random((n_classes, n_classes))
    logits = np.random.random((n_examples, n_classes))
    distances = distance_function(logits, mean_logits)
    assert type(distances) == np.ndarray
    assert distances.shape == (n_examples, n_classes)
