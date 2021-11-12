"""
Evaluation utilities for Openset-NLP experiments
"""

from typing import Callable, Tuple, List
from itertools import chain

from sklearn.metrics import roc_auc_score

from text_classification.dataset_utils import (
    InputMultilabelExample,
    OutputMultilabelExample
)


def compute_auc(y_true: List[float], y_score: List[float]) -> float:
    """
    compute AUC, requiring AUC to always be > 0.5
    """
    auc = roc_auc_score(y_true, y_score)
    auc = max(auc, 1 - auc)
    return auc


def incorrect_prediction_aucs(
        test_examples: List[InputMultilabelExample],
        prediction_examples: List[OutputMultilabelExample],
        positive_class_test: Callable[[float], bool],
        negative_class_test: Callable[[float], bool],
        save_plots: bool = False,
        filename_prefix: str = "") -> Tuple[float, float]:
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

    incorrect = positive_incorrect + negative_incorrect

    # compute positive vs. all AUC
    y_score = positive_correct + incorrect
    y_true = [0] * len(positive_correct) + [1] * len(incorrect)
    positive_class_auc = compute_auc(y_true, y_score)

    # compute negative vs. all AUC
    y_score = negative_correct + incorrect
    y_true = [0] * len(negative_correct) + [1] * len(incorrect)
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
        confidence_extraction_method: Callable,
        save_plots: bool = False,
        filename_prefix: str = "") -> float:
    """
    Compute AUC to determine how well we can distinguish between in-set
    vs. out-of-set (oos) examples using the associated confidences.
    """

    in_set_confidences = [confidence_extraction_method(example.confidences)
                          for example in in_set_prediction_examples]
    out_of_set_confidences = [confidence_extraction_method(example.confidences)
                              for example in out_of_set_prediction_examples]
    in_set_confidences = list(chain(*in_set_confidences))
    out_of_set_confidences = list(chain(*out_of_set_confidences))

    # compute AUC
    y_score = in_set_confidences + out_of_set_confidences
    y_true = [0] * len(in_set_confidences) + [1] * len(out_of_set_confidences)
    auc = compute_auc(y_true, y_score)

    if save_plots:
        confidences = [in_set_confidences, out_of_set_confidences]
        labels = ["in-set", "out-of-set"]
        out_filepath = "experiments/" + filename_prefix + "detect-oos.png"
        if filename_prefix.startswith("base-w-background"):
            model_name = "F+B"
        elif filename_prefix.startswith("base"):
            model_name = "F"
        else:
            model_name = "AA"
        title = "Model " + model_name
        confidence_histograms(confidences, labels, out_filepath, title=title)

    # compute per-example AUC
    agg_method = lambda confidences: sum([(c-0.5)**2 for c in confidences])
    in_set_confidences = [agg_method(example.confidences) for example in in_set_prediction_examples]
    out_of_set_confidences = [agg_method(example.confidences) for example in out_of_set_prediction_examples]
    y_score = in_set_confidences + out_of_set_confidences
    y_true = [0] * len(in_set_confidences) + [1] * len(out_of_set_confidences)
    agg_auc = compute_auc(y_true, y_score)

    return auc, agg_auc


def confidence_histograms(
        confidences: List[List[float]],
        labels: List[str],
        out_filepath: str,
        density: bool = False,
        title: str = "") -> None:
    """
    generate overlapping histograms of confidences and save to out_filepath
    """
    from matplotlib import pyplot as plt
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(10, 7))
    plt.style.use("tableau-colorblind10")
    for confidence, label in zip(confidences, labels):
        plt.hist(confidence, bins=30, label=label, density=density, alpha=0.3, histtype="stepfilled")
    plt.legend(loc="upper right")
    plt.xlabel("Class Membership Score")
    plt.ylabel("Count")
    plt.title(title)
    plt.savefig(out_filepath)
