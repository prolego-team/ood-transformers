"""
perform inference using a trained multi-label classifier
and display some metrics
"""


import click

from sklearn.metrics import confusion_matrix

from text_classification import (
    configs,
    inference_utils,
    eval_utils
)

from nlp_datasets import BACKGROUND_CATEGORIES, reuters_dataset_dictionaries, TOP_FIVE_CATEGORIES, reuters_dataset_to_train_test_examples


@click.command()
@click.argument("inference_config_filepath", type=click.Path(exists=True))
@click.option("--analyze_count", "-c", is_flag=True)
def main(**kwargs):
    # generate multi-label examples from test data
    _, test_examples = reuters_dataset_to_train_test_examples(
        categories=TOP_FIVE_CATEGORIES,
        background_categories=BACKGROUND_CATEGORIES,
        shuffle_train_examples=False
    )
    # test_examples = test_examples[:10]

    # read inference config
    inference_config = configs.read_config_for_inference(kwargs["inference_config_filepath"])

    # run inference
    predictor = inference_utils.MultilabelPredictor(
        inference_config.model_config,
        inference_config.class_labels
    )
    prediction_examples = predictor(test_examples, inference_config.max_length)

    # construct confusion matrices (one for each label)
    confusion_matrices = eval_utils.multilabel_confusion_matrix(
        test_examples, prediction_examples, inference_config.class_labels)

    for class_label, confusion_matrix in confusion_matrices.items():
        print(class_label)
        print(confusion_matrix)
        print()

    if kwargs["analyze_count"]:
        # construct confusion matrix for number of predictions predicted vs actual
        pred_count_true = [len(example.labels) for example in test_examples]
        pred_count_pred = [len(example.labels) for example in prediction_examples]
        print("Number of labels pred. vs. true")
        labels = sorted(set(pred_count_pred + pred_count_true))
        print(labels)
        print(confusion_matrix(pred_count_true, pred_count_pred, labels=labels))


if __name__ == "__main__":
    main()
