"""
Detect incorrect predictions using a model trained with objectosphere loss
"""


import os
from pprint import pprint

import click

from text_classification.configs import read_config_for_inference
from text_classification.inference_utils import MultilabelPredictor
from text_classification.eval_utils import multilabel_precision_recall

from eval import out_of_set_aucs
from nlp_datasets import (
    BACKGROUND_CATEGORIES,
    TOP_FIVE_CATEGORIES,
    movie_reviews_dataset_to_examples,
    reuters_dataset_to_train_test_examples
)


def run_inference_compute_performance(
        inference_config_filepath: str) -> dict:
    """
    Run inference on a trained model specified by inference_config_filepath
    using in-set positive ("foreground") and in-set negative ("background") examples.
    Compute precision, recall, and background accuracy.
    """
    # generate foreground/background multi-label examples from test data
    _, test_examples = reuters_dataset_to_train_test_examples(
        categories=TOP_FIVE_CATEGORIES,
        background_categories=BACKGROUND_CATEGORIES,
        shuffle_train_examples=False
    )
    foreground_test_examples = [e for e in test_examples if len(e.labels) > 0]
    background_test_examples = [e for e in test_examples if len(e.labels) == 0]
    # test_examples = test_examples[:10]

    # read inference config
    inference_config = read_config_for_inference(inference_config_filepath)

    # run inference on foreground and background examples
    predictor = MultilabelPredictor(
        inference_config.model_config,
        inference_config.class_labels
    )
    foreground_prediction_examples = predictor(
        foreground_test_examples, inference_config.max_length)
    background_prediction_examples = predictor(
        background_test_examples, inference_config.max_length)

    # compute precision and recall on foreground test set
    precision, recall = multilabel_precision_recall(
        foreground_test_examples,
        foreground_prediction_examples
    )

    # compute background accuracy (fraction of correctly identified background examples)
    # using the background test set
    pred_background = [e for e in background_prediction_examples
                       if len(e.labels) == 0]
    background_accuracy = len(pred_background) / len(background_prediction_examples)

    return {"precision": precision,
            "recall": recall,
            "background_accuracy": background_accuracy}


def run_inference_compute_auc(
        inference_config_filepath: str,
        plot_filename_prefix: str = "") -> dict:
    """
    run inference on in-set vs. out-of-set (both Reuters and Movie Reviews) data
    and compute AUCs
    """

    # Read inference config
    inference_config = read_config_for_inference(inference_config_filepath)

    # Load all reuters data, split into train/test/out-of-set
    train_examples, test_examples = reuters_dataset_to_train_test_examples(
        categories=None,
        background_categories=None,
        shuffle_train_examples=False
    )
    def contains_class_label(example, class_labels) -> bool:
        for label in example.labels:
            if label in class_labels:
                return True
        return False

    in_set_foreground_test = [e for e in test_examples
                              if contains_class_label(e, TOP_FIVE_CATEGORIES)]
    # in_set_background_test = [e for e in test_examples
    #                           if not contains_class_label(e, TOP_FIVE_CATEGORIES)
    #                           and contains_class_label(e, BACKGROUND_CATEGORIES)]
    # use data from train and test both for OOS
    oos_test = [e for e in (train_examples + test_examples)
                if not contains_class_label(e, TOP_FIVE_CATEGORIES + BACKGROUND_CATEGORIES)]

    # Load movie reviews data
    movie_reviews_examples = movie_reviews_dataset_to_examples(
        categories=TOP_FIVE_CATEGORIES
    )

    # Perform inference
    multilabel_predictor = MultilabelPredictor(
        inference_config.model_config,
        inference_config.class_labels
    )
    in_set_foreground_preds = multilabel_predictor(
        in_set_foreground_test, inference_config.max_length, -1.0)
    oos_preds = multilabel_predictor(
        oos_test, inference_config.max_length, -1.0
    )
    movies_preds = multilabel_predictor(
        movie_reviews_examples, inference_config.max_length, -1.0
    )

    # confidence_extraction_method = lambda confidences: confidences
    abs_distance = lambda confidences: [abs(c - 0.5) for c in confidences]
    squared_euclidean_distance = lambda confidences: sum([(c - 0.5) ** 2 for c in confidences])
    # Compute AUCs for in-set vs. out-of-set examples
    in_set_foreground_vs_oos = out_of_set_aucs(
        in_set_foreground_preds,
        oos_preds,
        confidence_transformation=abs_distance,
        confidence_aggregation=squared_euclidean_distance,
        save_plots=True,
        filename_prefix=plot_filename_prefix + "-reuters-"
    )
    in_set_foreground_vs_movies = out_of_set_aucs(
        in_set_foreground_preds,
        movies_preds,
        confidence_transformation=abs_distance,
        confidence_aggregation=squared_euclidean_distance,
        save_plots=True,
        filename_prefix=plot_filename_prefix + "-movies-"
    )

    # Format output
    out = {"Reuters AUC": in_set_foreground_vs_oos,
           "Movie Reviews AUC": in_set_foreground_vs_movies}
    return out


def generate_training_command(
        saved_model_dirpath: str,
        use_background_categories: bool,
        use_objectosphere_loss: bool) -> str:
    """
    General command line string for executing model training.
    """
    command = "python -m train_multilabel_classifier test_data/training_config.json "
    command += "-md " + saved_model_dirpath + " "
    command += "-icf " + os.path.join(saved_model_dirpath, "inference_config.json") + " "
    if use_background_categories:
        command += "-bc "
    if use_objectosphere_loss:
        command += "-ol "
    command = command.strip()
    return command


@click.command()
@click.option("--do_train", "-t", is_flag=True)
@click.option("--do_eval", "-e", is_flag=True)
@click.option("--do_auc", "-a", is_flag=True)
def main(**kwargs):
    """
    Main driver method for running model training (--do_train), performance
    evaluation on in-set data (--do_eval), and AUC on in-set vs. out-of-set
    data (--do_auc).
    """

    # Model training
    if kwargs["do_train"]:
        # base model, no background examples
        command = generate_training_command("trained_base", False, False)
        print(command)
        os.system(command)

        # base model, with background examples
        command = generate_training_command("trained_base_w_background", True, False)
        print(command)
        os.system(command)

        # objectosphere model, with background examples
        command = generate_training_command("trained_objectosphere", True, True)
        print(command)
        os.system(command)

    # Evaluate trained model performance on test set
    if kwargs["do_eval"]:
        # base model, no background examples
        inference_config_filepath = os.path.join("trained_base", "inference_config.json")
        base_out = run_inference_compute_performance(inference_config_filepath)
        print("Base")
        pprint(base_out)

        # base model, with background examples
        inference_config_filepath = os.path.join("trained_base_w_background", "inference_config.json")
        base_w_background_out = run_inference_compute_performance(inference_config_filepath)
        print("Base Trained with Background")
        pprint(base_w_background_out)

        # objectosphere model, with background examples
        inference_config_filepath = os.path.join("trained_objectosphere", "inference_config.json")
        objectosphere_out = run_inference_compute_performance(inference_config_filepath)
        print("Objectosphere")
        pprint(objectosphere_out)

    # Evaluate difference between in-set vs. out-of-set sample confidences
    if kwargs["do_auc"]:
        # Inference
        base_metrics = run_inference_compute_auc(
            "trained_base/inference_config.json",
            plot_filename_prefix="base")
        base_with_background_metrics = run_inference_compute_auc(
            "trained_base_w_background/inference_config.json",
            plot_filename_prefix="base-w-background")
        objectosphere_metrics = run_inference_compute_auc(
            "trained_objectosphere/inference_config.json",
            plot_filename_prefix="objectosphere")

        # Print Results
        print("Base")
        pprint(base_metrics)
        print("Base Trained with Background")
        pprint(base_with_background_metrics)
        print("Objectosphere")
        pprint(objectosphere_metrics)


if __name__ == "__main__":
    main()
