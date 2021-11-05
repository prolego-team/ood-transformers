"""
perform inference using a trained multi-label classifier
and display some metrics
"""


import click

from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

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
    foreground_test_examples = [e for e in test_examples if len(e.labels) > 0]
    background_test_examples = [e for e in test_examples if len(e.labels) == 0]
    # test_examples = test_examples[:10]

    # read inference config
    inference_config = configs.read_config_for_inference(kwargs["inference_config_filepath"])

    # run inference on foreground and background examples
    predictor = inference_utils.MultilabelPredictor(
        inference_config.model_config,
        inference_config.class_labels
    )
    foreground_prediction_examples = predictor(foreground_test_examples, inference_config.max_length)
    background_prediction_examples = predictor(background_test_examples, inference_config.max_length)

    # microaveraging precision/recall
    tp_count = 0
    fp_count = 0
    fn_count = 0
    for pred_example, true_example in zip(foreground_prediction_examples, foreground_test_examples):
        labels = list(set(pred_example.labels + true_example.labels))
        for label in labels:
            if label in true_example.labels:
                if label in pred_example.labels:
                    # true positive
                    tp_count += 1
                else:
                    # false negative
                    fn_count += 1
            else:
                # false positive
                fp_count += 1
    precision = tp_count / (tp_count + fp_count)
    recall = tp_count / (tp_count + fn_count)
    print("Precision:", precision)
    print("Recall:", recall)

    # compute percent of background examples correctly identified as such
    is_pred_background = [len(e.labels) == 0 for e in background_prediction_examples]
    print("Background examples correctly identified", len(is_pred_background), "/", len(background_prediction_examples))
    print(len(is_pred_background) / len(background_prediction_examples) * 100)

    if False:
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
        print(sklearn_confusion_matrix(pred_count_true, pred_count_pred, labels=labels))


if __name__ == "__main__":
    main()
