"""
Detect incorrect predictions using a model trained with objectosphere loss
"""


import os
from pprint import pprint

import click

from text_classification.configs import read_config_for_inference
from text_classification.inference_utils import MultilabelPredictor

from eval import out_of_set_aucs
from nlp_datasets import BACKGROUND_CATEGORIES, TOP_FIVE_CATEGORIES, movie_reviews_dataset_to_examples, reuters_dataset_to_train_test_examples


def run_inference_and_eval(inference_config_filepath: str, plot_filename_prefix: str = "") -> dict:
    """
    run inference on in-set vs. out-of-set (both Reuters and Movie Reviews) data
    and compute AUCs
    """

    # Read inference config
    inference_config = read_config_for_inference(inference_config_filepath)

    # Load reuters data, split into train/test/out-of-set
    _, test_examples = reuters_dataset_to_train_test_examples(
        categories=None,
        shuffle_train_examples=False
    )
    training_labels = TOP_FIVE_CATEGORIES + BACKGROUND_CATEGORIES
    def contains_class_label(example, class_labels) -> bool:
        for label in example.labels:
            if label in class_labels:
                return True
        return False

    in_set_foreground_test = [e for e in test_examples
                              if contains_class_label(e, TOP_FIVE_CATEGORIES)]
    in_set_background_test = [e for e in test_examples
                              if contains_class_label(e, BACKGROUND_CATEGORIES)]
    oos_test = [e for e in test_examples
                if not contains_class_label(e, TOP_FIVE_CATEGORIES + BACKGROUND_CATEGORIES)]

    # Load movie reviews data
    movie_reviews_examples = movie_reviews_dataset_to_examples(
        categories=training_labels
    )

    # Perform inference
    multilabel_predictor = MultilabelPredictor(
        inference_config.model_config,
        inference_config.class_labels
    )
    in_set_foreground_preds = multilabel_predictor(
        in_set_foreground_test, inference_config.max_length, 0.5)
    in_set_background_preds = multilabel_predictor(
        in_set_background_test, inference_config.max_length, 0.5
    )
    oos_preds = multilabel_predictor(
        oos_test, inference_config.max_length, 0.5
    )
    movies_preds = multilabel_predictor(
        movie_reviews_examples, inference_config.max_length, 0.5
    )

    # Compute AUCs for in-set vs. out-of-set examples
    in_set_foreground_vs_oos = out_of_set_aucs(
        in_set_foreground_preds,
        oos_preds,
        lambda confidences: max(confidences),
        save_plots=True,
        filename_prefix=plot_filename_prefix + "-reuters-foreground-"
    )
    in_set_background_vs_oos = out_of_set_aucs(
        in_set_background_preds,
        oos_preds,
        lambda confidences: max(confidences),
        save_plots=True,
        filename_prefix=plot_filename_prefix + "-reuters-background-"
    )
    in_set_foreground_vs_movies = out_of_set_aucs(
        in_set_foreground_preds,
        movies_preds,
        lambda confidences: max(confidences),
        save_plots=True,
        filename_prefix=plot_filename_prefix + "-movies-foreground-"
    )
    in_set_background_vs_movies = out_of_set_aucs(
        in_set_background_preds,
        movies_preds,
        lambda confidences: max(confidences),
        save_plots=True,
        filename_prefix="-movies-background-"
    )

    # Format output
    out = {"Reuters": {"in-set foreground vs. oos": in_set_foreground_vs_oos,
                       "in-set-background vs. oos": in_set_background_vs_oos},
           "Movie Reviews": {"in-set foreground vs. oos": in_set_foreground_vs_movies,
                             "in-set background vs. oos": in_set_background_vs_movies}}
    return out


def generate_training_command(
        saved_model_dirpath,
        use_background_categories,
        use_objectosphere_loss):
    command = "python -m train_multilabel_classifier test_data\training_config.json"
    command += "-md " + saved_model_dirpath + " "
    command += "-icf " + os.path.join(saved_model_dirpath, "inference_config.json") + " "
    if use_background_categories:
        command += "-bc "
    if use_objectosphere_loss:
        command += "-ol "
    command = command.strip()
    return command


@click.command()
@click.option("--do_train", "-dt", is_flag=True)
def main(**kwargs):
    # Model Training
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


    # Inference
    base_metrics = run_inference_and_eval("trained_base\inference_config.json", plot_filename_prefix="base")
    base_with_background_metrics = run_inference_and_eval("trained_base_w_background\inference_config.json", plot_filename_prefix="base-w-background")
    objectosphere_metrics = run_inference_and_eval("trained_objectosphere\inference_config.json", plot_filename_prefix="objectosphere")

    # Print Results
    print("Base")
    pprint(base_metrics)
    print("Base Trained with Background")
    pprint(base_with_background_metrics)
    print("Objectosphere")
    pprint(objectosphere_metrics)


if __name__ == "__main__":
    main()
