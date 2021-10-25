"""
Evaluation utilities for Openset-NLP experiments
"""

from typing import Callable, Tuple, List

from text_classification.dataset_utils import (
    InputMultilabelExample,
    OutputMultilabelExample
)

from experiment_utils import compute_auc


def incorrect_prediction_aucs(
        test_examples: List[InputMultilabelExample],
        prediction_examples: List[OutputMultilabelExample],
        positive_class_test: Callable[[float], bool],
        negative_class_test: Callable[[float], bool],
        save_plots: bool = False,
        filename_prefix: str = "",
        modified_auc: bool = False) -> Tuple[float, float]:
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

    if modified_auc:
        # compute positive vs. all AUC
        incorrect_or_negative = positive_incorrect + negative_incorrect + negative_correct
        y_score = positive_correct + incorrect_or_negative
        y_true = [1] * len(positive_correct) + [0] * len(incorrect_or_negative)
        positive_class_auc = compute_auc(y_true, y_score)

        # compute negative vs. all AUC
        incorrect_or_positive = positive_correct + positive_incorrect + negative_incorrect
        y_score = negative_correct + incorrect_or_positive
        y_true = [1] * len(negative_correct) + [0] * len(incorrect_or_positive)
        negative_class_auc = compute_auc(y_true, y_score)

    else:
        # compute positive class AUC
        y_score = positive_correct + positive_incorrect
        y_true = [0] * len(positive_correct) + [1] * len(positive_incorrect)
        positive_class_auc = compute_auc(y_true, y_score)

        # compute negative class AUC
        y_score = negative_correct + negative_incorrect
        y_true = [0] * len(negative_correct) + [1] * len(negative_incorrect)
        negative_class_auc = compute_auc(y_true, y_score)

    if save_plots:
        incorrect = positive_incorrect + negative_incorrect
        confidences = [positive_correct, incorrect, negative_correct]
        labels = ["positive correct", "incorrect", "negative correct"]
        out_filepath = "experiments/" + filename_prefix + "detect-incorrect.png"
        confidence_histograms(confidences, labels, out_filepath, density=True)

    return positive_class_auc, negative_class_auc


def out_of_set_aucs(
        in_set_prediction_examples: List[OutputMultilabelExample],
        out_of_set_prediction_examples: List[OutputMultilabelExample],
        positive_class_test: Callable[[float], bool],
        negative_class_test: Callable[[float], bool],
        save_plots: bool = False,
        filename_prefix: str = "",
        modified_auc: bool = False) -> Tuple[float, float]:
    """
    Compute AUCs to determine how well we can distinguish between in-set
    vs. out-of-set (oos) examples using the associated confidences.

    positive/negative_class_test is the test applied to a specific
    confidence score to determine whether it is a member of the positive/
    negative class
    """
    def group_confidences(
            prediction_examples: List[OutputMultilabelExample]
        ) -> Tuple[List[float], List[float]]:
        """
        group confidences into postive vs. negative class buckets:
        """
        positive_class_confidences = []
        negative_class_confidences = []
        for example in prediction_examples:
            for confidence in example.confidences:
                if positive_class_test(confidence):
                    positive_class_confidences.append(confidence)
                elif negative_class_test(confidence):
                    negative_class_confidences.append(confidence)
        return positive_class_confidences, negative_class_confidences

    in_set_positive, in_set_negative = group_confidences(in_set_prediction_examples)
    oos_positive, oos_negative = group_confidences(out_of_set_prediction_examples)

    if modified_auc:
        # positive vs. all
        oos_or_negative = oos_positive + oos_negative + in_set_negative
        y_score = in_set_positive + oos_or_negative
        y_true = [1] * len(in_set_positive) + [0] * len(oos_or_negative)
        positive_class_auc = compute_auc(y_true, y_score)

        # negative vs. all
        oos_or_positive = oos_positive + oos_negative + in_set_positive
        y_score = in_set_negative + oos_or_positive
        y_true = [1] * len(in_set_negative) + [0] * len(oos_or_positive)
        negative_class_auc = compute_auc(y_true, y_score)

    else:
        # compute positive class AUC
        y_score = oos_positive + in_set_positive
        y_true = [0] * len(oos_positive) + [1] * len(in_set_positive)
        positive_class_auc = compute_auc(y_true, y_score)

        # compute negative class AUC
        y_score = oos_negative + in_set_negative
        y_true = [0] * len(oos_negative) + [1] * len(in_set_negative)
        negative_class_auc = compute_auc(y_true, y_score)

    if save_plots:
        # positive class
        out_of_set = oos_negative + oos_positive
        confidences = [in_set_positive, out_of_set, in_set_negative]
        labels = ["in-set positive", "out-of-set", "in-set negative"]
        out_filepath = "experiments/" + filename_prefix + "detect-oos-positive.png"
        confidence_histograms(confidences, labels, out_filepath)

    return positive_class_auc, negative_class_auc


def confidence_histograms(
        confidences: List[List[float]],
        labels: List[str],
        out_filepath: str,
        density: bool = False) -> None:
    """
    generate overlapping histograms of confidences and save to out_filepath
    """
    from matplotlib import pyplot as plt
    plt.figure(figsize=(8, 6))
    for confidence, label in zip(confidences, labels):
        plt.hist(confidence, label=label, density=density, alpha=0.3)
    plt.legend(loc="upper right")
    plt.savefig(out_filepath)
