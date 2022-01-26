"""
BROS
Copyright 2022-present NAVER Corp.
Apache License v2.0
"""

import numpy as np
import torch
import torch.utils.data
from overrides import overrides

from lightning_modules.bros_module import BROSModule


class BROSSPADERELModule(BROSModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.eval_kwargs = {"dummy_idx": self.cfg.train.max_seq_length}

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


def do_eval_step(batch, head_outputs, loss, eval_kwargs):
    dummy_idx = eval_kwargs["dummy_idx"]

    el_outputs = head_outputs["el_outputs"]

    pr_el_labels = torch.argmax(el_outputs, -1)

    n_batch_gt_rel, n_batch_pr_rel, n_batch_correct_rel = eval_el_spade_batch(
        pr_el_labels,
        batch["el_labels"],
        batch["are_box_first_tokens"],
        dummy_idx,
    )

    step_out = {
        "loss": loss,
        "n_batch_gt_rel": n_batch_gt_rel,
        "n_batch_pr_rel": n_batch_pr_rel,
        "n_batch_correct_rel": n_batch_correct_rel,
    }

    return step_out


def eval_el_spade_batch(
    pr_el_labels,
    gt_el_labels,
    are_box_first_tokens,
    dummy_idx,
):
    n_batch_gt_rel, n_batch_pr_rel, n_batch_correct_rel = 0, 0, 0

    bsz = pr_el_labels.shape[0]
    for example_idx in range(bsz):
        n_gt_rel, n_pr_rel, n_correct_rel = eval_el_spade_example(
            pr_el_labels[example_idx],
            gt_el_labels[example_idx],
            are_box_first_tokens[example_idx],
            dummy_idx,
        )

        n_batch_gt_rel += n_gt_rel
        n_batch_pr_rel += n_pr_rel
        n_batch_correct_rel += n_correct_rel

    return n_batch_gt_rel, n_batch_pr_rel, n_batch_correct_rel


def eval_el_spade_example(pr_el_label, gt_el_label, box_first_token_mask, dummy_idx):
    gt_relations = parse_relations(gt_el_label, box_first_token_mask, dummy_idx)
    pr_relations = parse_relations(pr_el_label, box_first_token_mask, dummy_idx)

    n_gt_rel = len(gt_relations)
    n_pr_rel = len(pr_relations)
    n_correct_rel = len(gt_relations & pr_relations)

    return n_gt_rel, n_pr_rel, n_correct_rel


def parse_relations(el_label, box_first_token_mask, dummy_idx):
    valid_el_labels = el_label * box_first_token_mask
    valid_el_labels = valid_el_labels.cpu().numpy()
    el_label_np = el_label.cpu().numpy()

    valid_token_indices = np.where(
        ((valid_el_labels != dummy_idx) * (valid_el_labels != 0))
    )
    link_map_tuples = []
    for token_idx in valid_token_indices[0]:
        link_map_tuples.append((el_label_np[token_idx], token_idx))

    return set(link_map_tuples)


def do_eval_epoch_end(step_outputs):
    n_total_gt_rel, n_total_pred_rel, n_total_correct_rel = 0, 0, 0

    for step_out in step_outputs:
        n_total_gt_rel += step_out["n_batch_gt_rel"]
        n_total_pred_rel += step_out["n_batch_pr_rel"]
        n_total_correct_rel += step_out["n_batch_correct_rel"]

    precision = 0.0 if n_total_pred_rel == 0 else n_total_correct_rel / n_total_pred_rel
    recall = 0.0 if n_total_gt_rel == 0 else n_total_correct_rel / n_total_gt_rel
    f1 = (
        0.0
        if recall * precision == 0
        else 2.0 * recall * precision / (recall + precision)
    )

    scores = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    return scores
