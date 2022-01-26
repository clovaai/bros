"""
BROS
Copyright 2022-present NAVER Corp.
Apache License v2.0
"""

import os

import torch
import torch.utils.data
from overrides import overrides
from seqeval.metrics import f1_score, precision_score, recall_score

from lightning_modules.bros_module import BROSModule


class BROSBIESModule(BROSModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        label_map = get_label_map(self.cfg.dataset_root_path)
        self.eval_kwargs = {
            "ignore_index": self.ignore_index,
            "label_map": label_map,
        }

    @overrides
    def training_step(self, batch, batch_idx, *args):
        _, loss = self.net(batch)

        log_dict_input = {"train_loss": loss}
        self.log_dict(log_dict_input, sync_dist=True)
        return loss

    @torch.no_grad()
    @overrides
    def validation_step(self, batch, batch_idx, *args):
        head_outputs, loss = self.net(batch)
        step_out = do_eval_step(batch, head_outputs, loss, self.eval_kwargs)
        return step_out

    @torch.no_grad()
    @overrides
    def validation_epoch_end(self, validation_step_outputs):
        scores = do_eval_epoch_end(validation_step_outputs)
        self.print(
            f"precision: {scores['precision']:.4f}, recall: {scores['recall']:.4f}, f1: {scores['f1']:.4f}"
        )


def get_label_map(dataset_root_path):
    label_map_file = os.path.join(dataset_root_path, "labels.txt")
    label_map = {}
    lines = open(label_map_file, "r", encoding="utf-8").readlines()
    for line_idx, line in enumerate(lines):
        label_map[line_idx] = line.strip()
    return label_map


def do_eval_step(batch, head_outputs, loss, eval_kwargs):
    ignore_index = eval_kwargs["ignore_index"]
    label_map = eval_kwargs["label_map"]

    pr_labels = torch.argmax(head_outputs, -1).cpu().numpy()

    labels = batch["labels"]
    gt_labels = labels.cpu().numpy()

    prs, gts = [], []
    # https://github.com/microsoft/unilm/blob/master/layoutlm/deprecated/examples/seq_labeling/run_seq_labeling.py#L372
    bsz, max_seq_length = labels.shape
    for example_idx in range(bsz):
        example_prs, example_gts = [], []
        for token_idx in range(max_seq_length):
            if labels[example_idx, token_idx] != ignore_index:
                example_prs.append(label_map[pr_labels[example_idx, token_idx]])
                example_gts.append(label_map[gt_labels[example_idx, token_idx]])
        prs.append(example_prs)
        gts.append(example_gts)

    step_out = {
        "loss": loss,
        "prs": prs,
        "gts": gts,
    }

    return step_out


def do_eval_epoch_end(step_outputs):
    prs, gts = [], []
    for step_out in step_outputs:
        prs.extend(step_out["prs"])
        gts.extend(step_out["gts"])

    precision = precision_score(gts, prs)
    recall = recall_score(gts, prs)
    f1 = f1_score(gts, prs)

    scores = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    return scores
