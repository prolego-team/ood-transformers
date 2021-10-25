"""
Hypothesis:
   Different strategies for scoring predictions have different efficacies
   of distinguishing between in-set and out-of-set examples.
"""


from pprint import pprint

import click

from text_classification import configs
from text_classification.dataset_utils import InputMultilabelExample

import experiment_utils
import openmax
import eval
from nlp_datasets import (
    reuters_dataset_to_train_test_examples,
    movie_reviews_dataset_to_examples
)


@click.command()
@click.argument("inference_config_filepath", type=click.Path(exists=True))
@click.option("--distance_function", "-d", default="euclidean",
              help="euclidean, mae, fractional_euclidean, fractional_mae, member, or nonmember")
def main(**kwargs):

    # Read inference config
    inference_config = configs.read_config_for_inference(kwargs["inference_config_filepath"])

    # Load reuters data, split into train/test/out-of-set
    def contains_class_label(example: InputMultilabelExample) -> bool:
        """
        Returns True if at least one of the example labels is in the
        list of class_labels for the inference_config.
        Returns False otherwise.
        """
        for label in example.labels:
            if label in inference_config.class_labels:
                return True
        return False

    train_examples, test_examples = reuters_dataset_to_train_test_examples(
        categories=None,
        shuffle_train_examples=False)
    in_set_train_examples = [e for e in train_examples if contains_class_label(e)]
    in_set_test_examples = [e for e in test_examples if contains_class_label(e)]
    oos_train = [e for e in train_examples if not contains_class_label(e)]
    oos_test = [e for e in test_examples if not contains_class_label(e)]
    reuters_out_of_set_examples = oos_train + oos_test

    # Load movie reviews data
    movie_reviews_examples = movie_reviews_dataset_to_examples(
        categories=inference_config.class_labels)

    # Build wrapped predictors
    distance_function_map = {"euclidean": openmax.euclidean_distance_function,
                             "mae": openmax.mae_distance_function,
                             "fractional_euclidean": openmax.fractional_euclidean_distance_function,
                             "fractional_mae": openmax.fractional_absolute_distance_function,
                             "member": openmax.member_class_distance,
                             "nonmember": openmax.non_member_class_distance}
    out = experiment_utils.build_wrapped_predictors(
        inference_config,
        in_set_train_examples,
        openmax_distance_function=distance_function_map[kwargs["distance_function"]]
    )
    wrapped_multilabel_predictor, wrapped_openmax_predictor = out

    # Perform inference on in-set and out-of-set (oos) test data
    in_set_sigmoid_examples = wrapped_multilabel_predictor(in_set_test_examples)
    in_set_distance_examples = wrapped_openmax_predictor(in_set_test_examples)
    reuters_oos_sigmoid_examples = wrapped_multilabel_predictor(reuters_out_of_set_examples)
    reuters_oos_distance_examples = wrapped_openmax_predictor(reuters_out_of_set_examples)
    movies_oos_sigmoid_examples = wrapped_multilabel_predictor(movie_reviews_examples)
    movies_oos_distance_examples = wrapped_openmax_predictor(movie_reviews_examples)

    # Compute AUCs for in-set vs. out-of-set examples
    def auc_helper(
            out_of_set_sigmoid_examples,
            out_of_set_distance_examples,
            filename_prefix):
        # sigmoid
        sigmoid_auc = eval.out_of_set_aucs(
            in_set_sigmoid_examples,
            out_of_set_sigmoid_examples,
            lambda confidences: max(confidences),  # max score for sigmoid
            save_plots=True,
            filename_prefix=filename_prefix + "sigmoid-",
            modified_auc=True
        )
        # distance
        distance_auc = eval.out_of_set_aucs(
            in_set_distance_examples,
            out_of_set_distance_examples,
            lambda confidences: min(confidences),  # min score for distance
            save_plots=True,
            filename_prefix=filename_prefix + "distance-",
            modified_auc=True
        )
        out = {"sigmoid": sigmoid_auc,
               "distance": distance_auc}
        return out

    reuters_aucs = auc_helper(reuters_oos_sigmoid_examples, reuters_oos_distance_examples, "reuters")
    movies_aucs = auc_helper(movies_oos_sigmoid_examples, movies_oos_distance_examples, "movies")

    # print results
    print("Reuters")
    pprint(reuters_aucs)
    print("Movie Reviews")
    pprint(movies_aucs)


if __name__ == "__main__":
    main()
