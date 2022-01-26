"""
BROS
Copyright 2022-present NAVER Corp.
Apache License v2.0
"""

import torch
import torch.utils.data
from overrides import overrides

from lightning_modules.bros_module import BROSModule
from utils import get_class_names


class BROSBIOModule(BROSModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        class_names = get_class_names(self.cfg.dataset_root_path)
        self.eval_kwargs = {
            "class_names": class_names,
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


def do_eval_step(batch, head_outputs, loss, eval_kwargs):
    class_names = eval_kwargs["class_names"]

    pr_labels = torch.argmax(head_outputs, -1)

    n_batch_gt_classes, n_batch_pr_classes, n_batch_correct_classes = eval_ee_bio_batch(
        pr_labels,
        batch["bio_labels"],
        batch["are_box_first_tokens"],
        class_names,
    )

    step_out = {
        "loss": loss,
        "n_batch_gt_classes": n_batch_gt_classes,
        "n_batch_pr_classes": n_batch_pr_classes,
        "n_batch_correct_classes": n_batch_correct_classes,
    }

    return step_out


def eval_ee_bio_batch(pr_labels, gt_labels, are_box_first_tokens, class_names):
    n_batch_gt_classes, n_batch_pr_classes, n_batch_correct_classes = 0, 0, 0

    bsz = pr_labels.shape[0]
    for example_idx in range(bsz):
        n_gt_classes, n_pr_classes, n_correct_classes = eval_ee_bio_example(
            pr_labels[example_idx],
            gt_labels[example_idx],
            are_box_first_tokens[example_idx],
            class_names,
        )

        n_batch_gt_classes += n_gt_classes
        n_batch_pr_classes += n_pr_classes
        n_batch_correct_classes += n_correct_classes

    return (
        n_batch_gt_classes,
        n_batch_pr_classes,
        n_batch_correct_classes,
    )


def eval_ee_bio_example(pr_seq, gt_seq, box_first_token_mask, class_names):
    valid_gt_seq = gt_seq[box_first_token_mask]
    valid_pr_seq = pr_seq[box_first_token_mask]

    gt_parse = parse_from_seq(valid_gt_seq, class_names)
    pr_parse = parse_from_seq(valid_pr_seq, class_names)

    n_gt_classes, n_pr_classes, n_correct_classes = 0, 0, 0
    for class_idx in range(len(class_names)):
        # Evaluate by ID
        n_gt_classes += len(gt_parse[class_idx])
        n_pr_classes += len(pr_parse[class_idx])
        n_correct_classes += len(gt_parse[class_idx] & pr_parse[class_idx])

    return n_gt_classes, n_pr_classes, n_correct_classes


def parse_from_seq(seq, class_names):
    parsed = [[] for _ in range(len(class_names))]
    for i, label_id_tensor in enumerate(seq):
        label_id = label_id_tensor.item()

        if label_id == 0:  # O
            continue

        class_id = (label_id - 1) // 2
        is_b_tag = label_id % 2 == 1

        if is_b_tag:
            parsed[class_id].append((i,))
        elif len(parsed[class_id]) != 0:
            parsed[class_id][-1] = parsed[class_id][-1] + (i,)

    parsed = [set(indices_list) for indices_list in parsed]

    return parsed


def do_eval_epoch_end(step_outputs):
    n_total_gt_classes, n_total_pr_classes, n_total_correct_classes = 0, 0, 0

    for step_out in step_outputs:
        n_total_gt_classes += step_out["n_batch_gt_classes"]
        n_total_pr_classes += step_out["n_batch_pr_classes"]
        n_total_correct_classes += step_out["n_batch_correct_classes"]

    precision = (
        0.0 if n_total_pr_classes == 0 else n_total_correct_classes / n_total_pr_classes
    )
    recall = (
        0.0 if n_total_gt_classes == 0 else n_total_correct_classes / n_total_gt_classes
    )
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
