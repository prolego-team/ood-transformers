"""
Hypothesis:
   Different strategies for scoring predictions have different efficacies
   of distinguishing between correct predictions and incorrect predictions.
"""

import click

from text_classification import configs

from nlp_datasets import reuters_dataset_to_train_test_examples
import openmax
from eval import incorrect_prediction_aucs
import experiment_utils


@click.command()
@click.argument("inference_config_filepath", type=click.Path(exists=True))
def main(**kwargs):

    # Read inference config
    inference_config = configs.read_config_for_inference(kwargs["inference_config_filepath"])

    # Load data, split into train/test
    train_examples, test_examples = reuters_dataset_to_train_test_examples(
        categories=inference_config.class_labels,
        shuffle_train_examples=False)
    # train_examples = train_examples[:10]
    # test_examples = test_examples[:10]

    # Build wrapped predictors
    out = experiment_utils.build_wrapped_predictors(
        inference_config,
        train_examples,
        openmax_distance_function=openmax.euclidean_distance_function
    )
    wrapped_multilabel_predictor, wrapped_openmax_predictor = out

    # Use training data to derive thresholds for classification
    lower_bound_confidence = 5
    upper_bound_confidence = 95

    sigmoid_confidence_threshold = openmax.derive_confidence_threshold(
        train_examples,
        wrapped_multilabel_predictor,
        lower_bound_confidence,
        upper_bound_confidence
    )

    distance_confidence_threshold = openmax.derive_confidence_threshold(
        train_examples,
        wrapped_openmax_predictor,
        upper_bound_confidence,
        lower_bound_confidence
    )
    print("thresholds")
    print("sigma", sigmoid_confidence_threshold)
    print("distance", distance_confidence_threshold)

    # Perform inference on test data
    sigmoid_confidence_examples = wrapped_multilabel_predictor(test_examples)
    distance_confidence_examples = wrapped_openmax_predictor(test_examples)

    # compute AUCs for each
    sigmoid_positive_auc, sigmoid_negative_auc = incorrect_prediction_aucs(
        test_examples,
        sigmoid_confidence_examples,
        lambda confidence: confidence >= sigmoid_confidence_threshold,
        lambda confidence: confidence < sigmoid_confidence_threshold,
        save_plots=True,
        filename_prefix="sigmoid-"
    )
    distance_positive_auc, distance_negative_auc = incorrect_prediction_aucs(
        test_examples,
        distance_confidence_examples,
        lambda confidence: confidence <= distance_confidence_threshold,
        lambda confidence: confidence > distance_confidence_threshold,
        save_plots=True,
        filename_prefix="distance-"
    )

    # print results
    print("Sigmoid")
    print("   Positive")
    print("   ", round(sigmoid_positive_auc, 2))
    print("   Negative")
    print("   ", round(sigmoid_negative_auc, 2))
    print()
    print("OpenMax Distance")
    print("   Positive")
    print("   ", round(distance_positive_auc, 2))
    print("   Negative")
    print("   ", round(distance_negative_auc, 2))
    print()


if __name__ == "__main__":
    main()
