"""
Train a multi-label classifier to classify the Reuters data by topics
"""

import click

from text_classification import (
    configs,
    dataset_utils,
    model_utils,
    training_utils
)
from nlp_datasets import (
    reuters_dataset_dictionaries,
    reuters_class_labels
)


RANDOM_SEED = 12345

# see huggingface.co documentation for a full list of training arguments
# and what they mean:
# https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments
TRAINING_ARGUMENTS = {
    "do_train": True,
    "evaluation_strategy": "no",  # to enable evaluation during training, change to "steps"
    "logging_steps": 1,
    "num_train_epochs": 1.0,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 16,
    "do_predict": False,
    "block_size": 128,
    "seed": RANDOM_SEED,
    "gradient_accumulation_steps": 1,
    "save_strategy": "no",  # to save intermediate model checkpoints duing training, change to "steps" or "epochs",
    "weight_decay": 0  # make non-zero (e.g., 0.02) to apply regularization in AdamW optimizer
}


@click.command()
@click.argument("training_config_filepath", type=click.Path(exists=True))
@click.option("--inference_config_filepath", "-icf", default="inference_config.json",
              help="Path to save the inference config file created after training.")
@click.option("--do_class_weights", "-cw", is_flag=True,
              help="Weight the loss by relative class frequency to account for class imbalance.")
def main(**kwargs):
    """
    Train a transformers model to classify Reuters data by topic.
    """

    # read training config
    training_config = configs.read_config_for_training(kwargs["training_config_filepath"])
    TRAINING_ARGUMENTS["output_dir"] = training_config.model_config.saved_model_dirpath

    # read data and create train/test examples
    reuters_data = reuters_dataset_dictionaries()
    train_examples = dataset_utils.dictionaries_to_input_multilabel_examples(
        [d for d in reuters_data if d["is_train"]],
        False)
    test_examples = dataset_utils.dictionaries_to_input_multilabel_examples(
        [d for d in reuters_data if not d["is_train"]],
        False)
    train_examples = train_examples[:100]
    test_examples = test_examples[:100]

    # load tokenizer
    class_labels = reuters_class_labels()
    num_labels = len(class_labels)
    _, tokenizer = model_utils.load_pretrained_model_and_tokenizer(
        training_config.model_config,
        num_labels
    )

    # create train and test datasets
    train_dataset = dataset_utils.MultilabelDataset(
        train_examples,
        class_labels,
        tokenizer,
        TRAINING_ARGUMENTS["block_size"],
        predict=False
    )
    test_dataset = dataset_utils.MultilabelDataset(
        test_examples,
        class_labels,
        tokenizer,
        TRAINING_ARGUMENTS["block_size"],
        predict=False
    )

    # train model
    training_utils.train_multilabel_classifier(
        train_dataset,
        test_dataset,
        training_config.model_config,
        num_labels,
        TRAINING_ARGUMENTS,
        do_eval=True,
        do_class_weights=kwargs["do_class_weights"]
    )

    # create and save inference config
    trained_model_config = configs.ModelConfig(
        training_config.model_config.saved_model_dirpath,
        "main",
        None,
        "multilabel")
    inference_config = configs.InferenceConfig(
        trained_model_config, class_labels, TRAINING_ARGUMENTS["block_size"])
    configs.save_config_for_inference(inference_config, kwargs["inference_config_filepath"])


if __name__ == "__main__":
    main()
