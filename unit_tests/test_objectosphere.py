"""
unit tests for objectosphere.py
"""

import pytest

from transformers.hf_argparser import HfArgumentParser
from transformers.training_args import TrainingArguments

from text_classification.configs import ModelConfig
from text_classification.dataset_utils import InputMultilabelExample, MultilabelDataset
from text_classification.model_utils import load_pretrained_model_and_tokenizer
from text_classification.training_utils import build_compute_metrics

from ood_transformers import objectosphere

TRAINING_ARGUMENTS = {
    "do_train": True,
    "do_eval": True,
    "evaluation_strategy": "steps",  # to disable evaluation during training, change to "no"
    "logging_steps": 50,
    "num_train_epochs": 1.0,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 16,
    "do_predict": False,
    "block_size": 128,
    "seed": 12345,
    "gradient_accumulation_steps": 1,
    "save_strategy": "no",  # to save intermediate model checkpoints duing training, change to "steps" or "epochs",
    "weight_decay": 0,  # make non-zero (e.g., 0.02) to apply regularization in AdamW optimizer
    "output_dir": "./"
}

@pytest.mark.usefixtures("input_multilabel_examples")
@pytest.mark.usefixtures("class_labels")
def test_ObjectosphereTrainer(input_multilabel_examples, class_labels):
    """
    """
    # load model
    model_config = ModelConfig("roberta-base", "main", None, "multilabel")
    model, tokenizer = load_pretrained_model_and_tokenizer(model_config, 4)

    # create train/eval datasets
    background_multilabel_examples = [InputMultilabelExample(example.guid, example.text, [])
                                      for example in input_multilabel_examples]
    all_multilabel_examples = input_multilabel_examples + background_multilabel_examples
    train_dataset = MultilabelDataset(
        all_multilabel_examples,
        class_labels,
        tokenizer,
        max_length=128
    )
    eval_dataset = MultilabelDataset(
        all_multilabel_examples,
        class_labels,
        tokenizer,
        max_length=128
    )

    # set up trainer
    parser = HfArgumentParser(TrainingArguments)
    training_args = parser.parse_dict(TRAINING_ARGUMENTS)[0]
    trainer = objectosphere.ObjectosphereTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics(class_labels)
    )

    # train
    trainer.train()
