"""
Evaluation utilities for Openset-NLP experiments
"""

from typing import Tuple
from numpy import positive

from sklearn.metrics import roc_auc_score


def incorrect_prediction_aucs(
        test_examples,
        prediction_examples,
        positive_class_test,
        negative_class_test) -> Tuple[float, float]:
    """
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
                # positive
                if positive_class_test(confidence):
                    # correct
                    positive_correct.append(confidence)
                else:
                    positive_incorrect.append(confidence)
            else:
                # negative
                if negative_class_test(confidence):
                    # correct
                    negative_correct.append(confidence)
                else:
                    negative_incorrect.append(confidence)

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
