"""
Implementations related to OpenMax described in
A. Bendale, T. Boult “[Towards Open Set Deep Networks](http://vast.uccs.edu/~abendale/papers/0348.pdf)”
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016
"""

from typing import List, Union, Tuple, Dict, Callable
import shutil
from tempfile import mkdtemp

import numpy as np
import torch
from transformers.training_args import TrainingArguments
from sklearn.metrics import roc_auc_score

from text_classification import inference_utils, model_utils, configs
from text_classification.dataset_utils import (
    InputMultilabelExample,
    OutputMultilabelExample,
    MultilabelDataset
)


def euclidean_distance_function(
        logits: np.ndarray, mean_logits: np.ndarray) -> np.ndarray:
    """
    compute the euclidean distance between each logit and the
    set of mean logits
    """
    distances = []
    for logit in logits:
        distances += [np.linalg.norm(mean_logits - logit, axis=1)]
    return np.array(distances)


def mae_distance_function(
        logits: np.ndarray, mean_logits: np.ndarray
    ) -> np.ndarray:
    """
    mean absolute error distance between each logit and the
    set of mean logits
    (note: this is actually the sum, not the mean)
    """
    distances = []
    for logit in logits:
        distances += [np.sum(np.abs(logit - mean_logits), axis=1)]
    return np.array(distances)


def fractional_absolute_distance_function(
        logits: np.ndarray, mean_logits: np.ndarray
    ) -> np.ndarray:
    """
    compute distance as the fractional change in the logit
    values with respect to mean logits
    """
    distances = []
    for logit in logits:
        fractional_change = (logit - mean_logits) / mean_logits
        distances += [np.sum(np.abs(fractional_change), axis=1)]
    return np.array(distances)


def fractional_euclidean_distance_function(
        logits: np.ndarray, mean_logits: np.ndarray
    ) -> np.ndarray:
    """
    compute distance as the fractional change in the logit
    values with respect to the mean logits, then normalize
    """
    distances = []
    for logit in logits:
        fractional_change = (logit - mean_logits) / mean_logits
        distances += [np.linalg.norm(fractional_change, axis=1)]
    return np.array(distances)


class OpenMaxPredictor(inference_utils.MultilabelPredictor):
    """
    Predict and return the output of the penultimate layer
    """
    def __init__(
            self,
            model_config,
            class_list,
            mean_logits: Dict[str, np.ndarray],
            distance_function: Callable = euclidean_distance_function) -> None:
        super().__init__(model_config, class_list)
        self.mean_logits = mean_logits
        self.distance_function = distance_function

    def predict_proba(self, test_dataset: MultilabelDataset) -> np.ndarray:
        """
        override predict_proba from MultilabelPredictor to return
        logits
        """

        # set up trainer
        temp_dir = mkdtemp()
        training_args = TrainingArguments(
            output_dir=temp_dir,
            do_train=False,
            do_eval=False,
            do_predict=True
        )
        trainer = model_utils.MultilabelTrainer(
            model=self.model,
            args=training_args
        )
        trainer.set_class_weights(None)

        # make predictions
        predictions = trainer.predict(test_dataset=test_dataset).predictions

        # clean up
        shutil.rmtree(temp_dir)

        torch.cuda.empty_cache()
        return predictions

    def confidences_to_predicted_labels(
            self,
            logits: np.array,
            threshold: Union[float, List[float]]) -> Tuple[List[List[str]], List[List[float]]]:
        """
        compute distance metric between logits and self.mean_logits
        if distance threshold exceeds threshold, assign prediction
        """
        mean_logits_array = np.array([self.mean_logits[k] for k in self.class_list])
        distances = self.distance_function(logits, mean_logits_array)

        if type(threshold) == float:
            threshold = [threshold] * len(self.class_list)
        one_hot_predictions = (np.array(distances) < threshold).astype(int).tolist()

        # convert predictions to labels
        index_to_label_mapping = {i: lab for i, lab in enumerate(self.class_list)}
        prediction_indices = inference_utils.one_hot_to_index_labels(one_hot_predictions)
        predicted_labels = [[index_to_label_mapping[index] for index in indices] for indices in prediction_indices]

        # extract confidences for labels
        def extract_positive_class_distances(row_distances, indices):
            return list(row_distances[indices])
        predicted_confidences = [extract_positive_class_distances(row, indices)
                                 for row, indices in zip(distances, prediction_indices)]

        return predicted_labels, predicted_confidences

    def __call__(
            self,
            examples,
            max_length,
            threshold
        ):
        return super().__call__(examples, max_length, threshold)


    # def confidences_to_predicted_labels(
            # self,
            # logits: np.array,
            # threshold: Union[float, List[float]]) -> Tuple[List[List[str]], List[List[float]]]:
        # """
        # apply sigmoid activation to logits before running confidences_to_predicted_labels
        # """
        # confidences = torch.sigmoid(torch.tensor(logits)).numpy()
        # return super().confidences_to_predicted_labels(confidences, threshold)


def examples_to_mean_logit(
        examples: List[InputMultilabelExample],
        inference_config: configs.InferenceConfig) -> np.ndarray:
    """
    run inference to extract logits for examples,
    compute the mean logit across examples
    """

    # run inference to extract logits
    predictor = OpenMaxPredictor(
        inference_config.model_config,
        inference_config.class_labels,
        mean_logits={})
    dataset = predictor.create_dataset(examples, inference_config.max_length)
    logits = predictor.predict_proba(dataset)
    return np.mean(logits, axis=0)


def examples_to_mean_logits(
    examples: List[InputMultilabelExample],
    inference_config: configs.InferenceConfig) -> Dict[str, np.ndarray]:
    """
    run inference to extract logits for each example
    construct dictionary of mean logits for each category
    """

    mean_logits = {}
    for class_label in inference_config.class_labels:
        class_examples = [e for e in examples if class_label in e.labels]
        class_mean_logit = examples_to_mean_logit(class_examples, inference_config)
        mean_logits[class_label] = class_mean_logit
    return mean_logits


def derive_confidence_threshold(
        examples: List[InputMultilabelExample],
        wrapped_predictor: Callable[[List[InputMultilabelExample]],
                                    List[OutputMultilabelExample]],
        positive_class_percentile: int = 5,
        negative_class_percentile: int = 95) -> float:
    """
    Identify a threshold on confidence values to determine
    whether an example should be classified as a member of the positive
    class.
    """
    # run examples through predictor
    predicted_examples = wrapped_predictor(
        examples
    )

    # separate confidences into positive and negative classes
    positive_confidences = []
    negative_confidences = []
    for predicted_example, example in zip(predicted_examples, examples):
        for confidence, label in zip(predicted_example.confidences,
                                     predicted_example.labels):
            if label in example.labels:
                # positive
                positive_confidences.append(confidence)
            else:
                # negative
                negative_confidences.append(confidence)

    # compute percentiles
    positive_percentile = np.percentile(
        positive_confidences, positive_class_percentile)
    negative_percentile = np.percentile(
        negative_confidences, negative_class_percentile)

    # return the average
    return (positive_percentile + negative_percentile) * 0.5
