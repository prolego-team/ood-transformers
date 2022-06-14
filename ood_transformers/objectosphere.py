"""
utilites related to objectosphere approach for reducing network agnostophobia
described in:

Reducing Network Agnostophobia
Akshay Raj Dhamija, Manuel Günther, Terrance E. Boult
https://arxiv.org/abs/1811.04110
"""

from typing import Tuple
from copy import deepcopy

import numpy as np
import torch
from transformers.modeling_outputs import SequenceClassifierOutput

from text_classification.model_utils import MultilabelTrainer


def split_inputs(inputs: dict) -> Tuple[dict, dict]:
    """
    Split inputs into foreground and background inputs.
    Background inputs are identified as those that have no
       non-zero labels.
    """
    # construct foreground and background inputs
    input_ids = inputs["input_ids"].cpu().numpy()
    attention_mask = inputs["attention_mask"].cpu().numpy()
    labels = inputs["labels"].cpu().numpy()

    background = {k: [] for k in ["input_ids", "attention_mask", "labels"]}
    foreground = {k: [] for k in ["input_ids", "attention_mask", "labels"]}

    for ids, mask, lab in zip(input_ids, attention_mask, labels):
        # foreground inputs have at least 1 non-zero label
        if np.any(lab > 0):
            foreground["input_ids"].append(ids)
            foreground["attention_mask"].append(mask)
            foreground["labels"].append(lab)
        else:
            background["input_ids"].append(ids)
            background["attention_mask"].append(mask)
            background["labels"].append(lab)
    foreground = {k: torch.IntTensor(v) for k, v in foreground.items()}
    background = {k: torch.IntTensor(v) for k, v in background.items()}
    if torch.cuda.device_count() > 0:
        foreground = {k: v.to(device="cuda") for k, v in foreground.items()}
        background = {k: v.to(device="cuda") for k, v in background.items()}
    return foreground, background


class ObjectosphereTrainer(MultilabelTrainer):

    def compute_background_loss(self, model, inputs):
        """
        Compute MSELoss against a vector of [0 0 0 0]s for background
           examples.
        This has the effect of driving all logits to 0.
        """
        inputs_copy = deepcopy(inputs)
        labels = inputs_copy.pop("labels")
        outputs = model(**inputs_copy)
        logits = outputs.logits
        loss_function = torch.nn.MSELoss()
        loss = loss_function(logits.view(-1, self.model.config.num_labels),
                             labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss separately for foreground a background examples.
        For foreground, use the same loss as MultilabelTrainer (binary
           cross entropy loss).
        For background, use the loss defined in compute_background_loss (
            mean squared error).
        Accumulate foreground and background losses.
        """

        # split inputs into foreground vs. background
        foreground_inputs, background_inputs = split_inputs(inputs)

        # check if foreground and/or background examples exist in the
        # current inputs
        compute_foreground = len(foreground_inputs["input_ids"]) > 0
        compute_background = len(background_inputs["input_ids"]) > 0

        # foreground loss
        self.class_weights = None
        if compute_foreground:
            foreground_loss, foreground_outputs = super().compute_loss(model, foreground_inputs, return_outputs=True)

        # background loss
        if compute_background:
            background_loss, background_outputs = self.compute_background_loss(model, background_inputs)

        if compute_foreground and compute_background:
            # accumulate
            loss = foreground_loss + background_loss
            #outputs = SequenceClassifierOutput(loss=loss, logits=torch.cat((background_outputs.logits, foreground_outputs.logits), dim=0))
            outputs = torch.cat((background_outputs.logits, foreground_outputs.logits))
            # outputs = background_outputs | foreground_outputs
        elif compute_foreground:
            loss = foreground_loss
            outputs = foreground_outputs
        elif compute_background:
            loss = background_loss
            outputs = background_outputs

        if return_outputs:
            return (loss, outputs)
        else:
            return loss
