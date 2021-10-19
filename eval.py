"""
Evaluation utilities for Openset-NLP experiments
"""

from typing import Callable, Tuple, List

from sklearn.metrics import roc_auc_score

from text_classification.dataset_utils import (
    InputMultilabelExample,
    OutputMultilabelExample
)


def incorrect_prediction_aucs(
        test_examples: List[InputMultilabelExample],
        prediction_examples: List[OutputMultilabelExample],
        positive_class_test: Callable[[float], bool],
        negative_class_test: Callable[[float], bool]) -> Tuple[float, float]:
    """
    Compute AUCs to determine how well we can distiguish between correct
    vs. incorrect predictions using the associated confidences.

    positive/negative_class_test is the test applied to a specific
    confidence score to determine whether it is a member of the positive/
    negative class
    """
    # sort confidences into buckets:
        # - positive class, correct label
        # - positive class, incorrect label
        # - negative class, correct label
        # - negative class, incorrect label
    positive_correct = []
    positive_incorrect = []
    negative_correct = []
    negative_incorrect = []
    for test_example, pred_example in zip(test_examples, prediction_examples):
        for confidence, label in zip(pred_example.confidences, pred_example.labels):
            if label in test_example.labels:
                if positive_class_test(confidence):
                    # positive correct
                    positive_correct.append(confidence)
                else:
                    # negative incorrect
                    negative_incorrect.append(confidence)
            else:
                if negative_class_test(confidence):
                    # negative correct
                    negative_correct.append(confidence)
                else:
                    # positive incorrect
                    positive_incorrect.append(confidence)

    # compute positive class AUC
    y_score = positive_correct + positive_incorrect
    y_true = [0] * len(positive_correct) + [1] * len(positive_incorrect)
    positive_class_auc = roc_auc_score(y_true, y_score)
    positive_class_auc = max(positive_class_auc, 1 - positive_class_auc)

    # compute negative class AUC
    y_score = negative_correct + negative_incorrect
    y_true = [0] * len(negative_correct) + [1] * len(negative_incorrect)
    negative_class_auc = roc_auc_score(y_true, y_score)
    negative_class_auc = max(negative_class_auc, 1 - negative_class_auc)

    return positive_class_auc, negative_class_auc
