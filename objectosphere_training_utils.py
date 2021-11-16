"""
Utilities for running model training.

Copied from text_classification.training_utils, but modified to allow
a custom multilabel_trainer to be passed as input.
"""

import os
from typing import Dict, Optional

import torch
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed
)

from text_classification import configs, dataset_utils, model_utils
from text_classification.training_utils import compute_multilabel_accuracy


def train_multilabel_classifier(
        train_dataset: dataset_utils.MultilabelDataset,
        eval_dataset: Optional[dataset_utils.MultilabelDataset],
        model_config: configs.ModelConfig,
        num_labels: int,
        training_arguments: dict,
        use_fast: bool = model_utils.USE_FAST_TOKENIZER,
        do_eval: bool = True,
        do_class_weights: bool = False,
        multilabel_trainer = model_utils.MultilabelTrainer) -> None:
    """
    Training loop for multi-label classification
    Copied from text_classification.training_utils, but modified to allow
       a custom multilabel_trainer to be passes as input.
    """

    if do_class_weights:
        class_weights = dataset_utils.compute_class_weights(train_dataset.labels)
    else:
        class_weights = None

    # build training args
    do_eval = (do_eval) and (eval_dataset is not None)  # if eval_dataset is None, set do_eval to False
    training_arguments["do_eval"] = do_eval  # do evaluation every logging_steps steps
    training_arguments["report_to"] = "none"
    parser = HfArgumentParser(TrainingArguments)
    training_args = parser.parse_dict(training_arguments)[0]

    # set seed for training
    set_seed(training_args.seed)

    # load model and tokenizer
    model, tokenizer = model_utils.load_pretrained_model_and_tokenizer(
        model_config, num_labels, use_fast)

    # set up trainer
    trainer = multilabel_trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_multilabel_accuracy
    )
    trainer.set_class_weights(class_weights)

    # train and save model and tokenizer
    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    # eval and write results to txt file
    if do_eval:
        eval_results = trainer.evaluate()
        eval_filepath = os.path.join(training_args.output_dir, "eval_results.txt")
        with open(eval_filepath, "w") as f:
            for k, v in sorted(eval_results.items()):
                f.write("%s = %s\n" % (k, str(v)))

    # release GPU memory
    torch.cuda.empty_cache()
