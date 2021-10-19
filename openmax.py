"""
Implementations related to OpenMax described in
A. Bendale, T. Boult “[Towards Open Set Deep Networks](http://vast.uccs.edu/~abendale/papers/0348.pdf)”
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016
"""

from typing import List, Union, Tuple, Dict
import shutil
from tempfile import mkdtemp

import numpy as np
import torch
from transformers.training_args import TrainingArguments

from text_classification import inference_utils
from text_classification import dataset_utils, model_utils, configs


def euclidean_distance_function(logits, mean_logits):
    """
    compute the euclidean distance between each logit and the
    set of mean logits
    """
    distances = []
    for logit in logits:
        distances += np.linalg.norm(mean_logits - logit, axis=1)
    return np.array(distances)


class OpenMaxPredictor(inference_utils.MultilabelPredictor):
    """
    Predict and return the output of the penultimate layer
    """
    def __init__(self, model_config, class_list, mean_logits: Dict[str, np.ndarray]):
        super().__init__(model_config, class_list)
        self.mean_logits = mean_logits

    def predict_proba(self, test_dataset: dataset_utils.MultilabelDataset) -> np.ndarray:
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
        distances = []
        for logit in logits:
            distances += [np.linalg.norm(mean_logits_array - logit, axis=1)]

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
            threshold,
            distance_function=euclidean_distance_function
        ):
        self.distance_function = distance_function
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
        examples: List[dataset_utils.InputMultilabelExample],
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
