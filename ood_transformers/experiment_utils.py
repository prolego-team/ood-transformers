"""
utility functions for running experiments
"""

from typing import Tuple, Callable, List

import numpy as np

from text_classification.inference_utils import MultilabelPredictor
from text_classification.dataset_utils import InputMultilabelExample, OutputMultilabelExample

from . import openmax


def build_wrapped_predictors(
        inference_config,
        train_examples,
        openmax_distance_function: Callable = openmax.euclidean_distance_function
    ) -> Tuple[Callable[[List[InputMultilabelExample]], List[OutputMultilabelExample]],
               Callable[[List[InputMultilabelExample]], List[OutputMultilabelExample]]]:
    """
    Construct wrapped multilabel predictor and openmax predictor
    Note: train_examples are used to compute mean logits for openmax predictor
    """

    # Build predictors
    multilabel_predictor = MultilabelPredictor(
        inference_config.model_config,
        inference_config.class_labels
    )
    mean_logits = openmax.examples_to_mean_logits(
        train_examples, inference_config)
    openmax_predictor = openmax.OpenMaxPredictor(
        inference_config.model_config,
        inference_config.class_labels,
        mean_logits=mean_logits,
        distance_function=openmax_distance_function  # TODO: test out other distance functions
    )

    # Wrap
    wrapped_multilabel_predictor = lambda examples: multilabel_predictor(
        examples, inference_config.max_length, 0.0
    )
    wrapped_openmax_predictor = lambda examples: openmax_predictor(
        examples, inference_config.max_length, np.inf
    )

    return wrapped_multilabel_predictor, wrapped_openmax_predictor
