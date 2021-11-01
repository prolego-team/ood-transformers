"""
Utilities for running model training
"""

import os
from typing import Dict, Optional

import numpy as np
import torch
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    EvalPrediction,
    set_seed
)

from text_classification import configs, dataset_utils, model_utils


def compute_multilabel_accuracy(
        prediction: EvalPrediction,
        threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute accuracy of one-hot encoded representation of predictions
    accuracy = # of correct predictions / # of total predictions
    """
    # apply activation
    scores = torch.sigmoid(torch.tensor(prediction.predictions)).numpy()
    preds = (scores > threshold).astype(int)
    return {"accuracy": np.sum(preds == prediction.label_ids) / (preds.shape[0] * preds.shape[1])}


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
    training loop for multi-label classification
    Notes:
       training_arguments: "output_dir" is a required key in the dictionary
       do_eval: if True, evaluation during training (every logging_steps steps)
          and at the end of training is performed. Note that if eval_dataset is None,
          do_eval is automatically reset to False
    """

    if do_class_weights:
        class_weights = dataset_utils.compute_class_weights(train_dataset.labels)
    else:
        class_weights = None

    # build training args
    do_eval = (do_eval) and (eval_dataset is not None)  # if eval_dataset is None, set do_eval to False
    training_arguments["do_eval"] = do_eval  # do evaluation every logging_steps steps
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
