"""
Train a multi-label classifier to classify the Reuters data by topics
"""

import os

import click
import pickle

from text_classification import (
    configs,
    dataset_utils,
    model_utils,
    training_utils
)
from nlp_datasets import (
    TOP_FIVE_CATEGORIES,
    BACKGROUND_CATEGORIES,
    reuters_dataset_to_train_test_examples
)
import openmax, objectosphere


RANDOM_SEED = 12345

# see huggingface.co documentation for a full list of training arguments
# and what they mean:
# https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments
TRAINING_ARGUMENTS = {
    "do_train": True,
    "evaluation_strategy": "steps",  # to disable evaluation during training, change to "no"
    "logging_steps": 50,
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
@click.option("--use_background_categories", "-bc", is_flag=True)
@click.option("--saved_model_dirpath", "-md", default=None)
@click.option("--use_objectosphere_loss", "-ol", is_flag=True)
def main(**kwargs):
    """
    Train a transformers model to classify Reuters data by topic.
    """

    # read training config
    training_config = configs.read_config_for_training(kwargs["training_config_filepath"])
    if kwargs["saved_model_dirpath"]:
        TRAINING_ARGUMENTS["output_dir"] = kwargs["saved_model_dirpath"]
    else:
        TRAINING_ARGUMENTS["output_dir"] = training_config.model_config.saved_model_dirpath

    # read data and create train/test examples
    if kwargs["use_background_categories"]:
        categories = TOP_FIVE_CATEGORIES + BACKGROUND_CATEGORIES
    else:
        categories = TOP_FIVE_CATEGORIES
    train_examples, test_examples = reuters_dataset_to_train_test_examples(
        categories=categories,
        shuffle_train_examples=True,
        seed=RANDOM_SEED
    )

    # train_examples = train_examples[:100]
    # test_examples = test_examples[:100]

    # load tokenizer
    num_labels = len(TOP_FIVE_CATEGORIES)
    _, tokenizer = model_utils.load_pretrained_model_and_tokenizer(
        training_config.model_config,
        num_labels
    )

    # create train and test datasets
    train_dataset = dataset_utils.MultilabelDataset(
        train_examples,
        TOP_FIVE_CATEGORIES,
        tokenizer,
        TRAINING_ARGUMENTS["block_size"],
        predict=False
    )
    test_dataset = dataset_utils.MultilabelDataset(
        test_examples,
        TOP_FIVE_CATEGORIES,
        tokenizer,
        TRAINING_ARGUMENTS["block_size"],
        predict=False
    )

    # train model
    if kwargs["use_objectosphere_loss"]:
        multilabel_trainer = objectosphere.ObjectosphereTrainer
    else:
        multilabel_trainer = model_utils.MultilabelTrainer
    training_utils.train_multilabel_classifier(
        train_dataset,
        test_dataset,
        training_config.model_config,
        num_labels,
        TRAINING_ARGUMENTS,
        do_eval=True,
        do_class_weights=kwargs["do_class_weights"],
        multilabel_trainer=multilabel_trainer
    )

    # create and save inference config
    trained_model_config = configs.ModelConfig(
        training_config.model_config.saved_model_dirpath,
        "main",
        None,
        "multilabel")
    inference_config = configs.InferenceConfig(
        trained_model_config, TOP_FIVE_CATEGORIES, TRAINING_ARGUMENTS["block_size"])
    configs.save_config_for_inference(inference_config, kwargs["inference_config_filepath"])

    # construct mean vectors from training examples and save
    mean_logits = openmax.examples_to_mean_logits(train_examples, inference_config)
    mean_logits_filepath = os.path.join(training_config.model_config.saved_model_dirpath, "mean_logits.pkl")
    with open(mean_logits_filepath, "wb") as f:
        pickle.dump(mean_logits, f)


if __name__ == "__main__":
    main()
