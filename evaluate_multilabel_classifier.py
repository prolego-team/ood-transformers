"""
perform inference using a trained multi-label classifier
and display some metrics
"""


import click

from text_classification import (
    configs,
    inference_utils,
    dataset_utils,
    eval_utils
)

from nlp_datasets import reuters_dataset_dictionaries, TOP_FIVE_CATEGORIES


@click.command()
@click.argument("inference_config_filepath", type=click.Path(exists=True))
def main(**kwargs):
    # generate multi-label examples from test data
    reuters_data = reuters_dataset_dictionaries(categories=TOP_FIVE_CATEGORIES)
    test_examples = dataset_utils.dictionaries_to_input_multilabel_examples(
        [d for d in reuters_data if not d["is_train"]],
        False)
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

if __name__ == "__main__":
    main()
