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
from utils import get_class_names


class BROSSPADEModule(BROSModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        class_names = get_class_names(self.cfg.dataset_root_path)
        self.eval_kwargs = {
            "class_names": class_names,
            "dummy_idx": self.cfg.train.max_seq_length,
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
    dummy_idx = eval_kwargs["dummy_idx"]

    itc_outputs = head_outputs["itc_outputs"]
    stc_outputs = head_outputs["stc_outputs"]

    pr_itc_labels = torch.argmax(itc_outputs, -1)
    pr_stc_labels = torch.argmax(stc_outputs, -1)

    (
        n_batch_gt_classes,
        n_batch_pr_classes,
        n_batch_correct_classes,
    ) = eval_ee_spade_batch(
        pr_itc_labels,
        batch["itc_labels"],
        batch["are_box_first_tokens"],
        pr_stc_labels,
        batch["stc_labels"],
        batch["attention_mask"],
        class_names,
        dummy_idx,
    )

    step_out = {
        "loss": loss,
        "n_batch_gt_classes": n_batch_gt_classes,
        "n_batch_pr_classes": n_batch_pr_classes,
        "n_batch_correct_classes": n_batch_correct_classes,
    }

    return step_out


def eval_ee_spade_batch(
    pr_itc_labels,
    gt_itc_labels,
    are_box_first_tokens,
    pr_stc_labels,
    gt_stc_labels,
    attention_mask,
    class_names,
    dummy_idx,
):
    n_batch_gt_classes, n_batch_pr_classes, n_batch_correct_classes = 0, 0, 0

    bsz = pr_itc_labels.shape[0]
    for example_idx in range(bsz):
        n_gt_classes, n_pr_classes, n_correct_classes = eval_ee_spade_example(
            pr_itc_labels[example_idx],
            gt_itc_labels[example_idx],
            are_box_first_tokens[example_idx],
            pr_stc_labels[example_idx],
            gt_stc_labels[example_idx],
            attention_mask[example_idx],
            class_names,
            dummy_idx,
        )

        n_batch_gt_classes += n_gt_classes
        n_batch_pr_classes += n_pr_classes
        n_batch_correct_classes += n_correct_classes

    return (
        n_batch_gt_classes,
        n_batch_pr_classes,
        n_batch_correct_classes,
    )


def eval_ee_spade_example(
    pr_itc_label,
    gt_itc_label,
    box_first_token_mask,
    pr_stc_label,
    gt_stc_label,
    attention_mask,
    class_names,
    dummy_idx,
):
    gt_first_words = parse_initial_words(
        gt_itc_label, box_first_token_mask, class_names
    )
    gt_class_words = parse_subsequent_words(
        gt_stc_label, attention_mask, gt_first_words, dummy_idx
    )

    pr_init_words = parse_initial_words(pr_itc_label, box_first_token_mask, class_names)
    pr_class_words = parse_subsequent_words(
        pr_stc_label, attention_mask, pr_init_words, dummy_idx
    )

    n_gt_classes, n_pr_classes, n_correct_classes = 0, 0, 0
    for class_idx in range(len(class_names)):
        # Evaluate by ID
        gt_parse = set(gt_class_words[class_idx])
        pr_parse = set(pr_class_words[class_idx])

        n_gt_classes += len(gt_parse)
        n_pr_classes += len(pr_parse)
        n_correct_classes += len(gt_parse & pr_parse)

    return n_gt_classes, n_pr_classes, n_correct_classes


def parse_initial_words(itc_label, box_first_token_mask, class_names):
    itc_label_np = itc_label.cpu().numpy()
    box_first_token_mask_np = box_first_token_mask.cpu().numpy()

    outputs = [[] for _ in range(len(class_names))]
    for token_idx, label in enumerate(itc_label_np):
        if box_first_token_mask_np[token_idx] and label != 0:
            outputs[label].append(token_idx)

    return outputs


def parse_subsequent_words(stc_label, attention_mask, init_words, dummy_idx):
    max_connections = 50

    valid_stc_label = stc_label * attention_mask.bool()
    valid_stc_label = valid_stc_label.cpu().numpy()
    stc_label_np = stc_label.cpu().numpy()

    valid_token_indices = np.where(
        (valid_stc_label != dummy_idx) * (valid_stc_label != 0)
    )

    next_token_idx_dict = {}
    for token_idx in valid_token_indices[0]:
        next_token_idx_dict[stc_label_np[token_idx]] = token_idx

    outputs = []
    for init_token_indices in init_words:
        sub_outputs = []
        for init_token_idx in init_token_indices:
            cur_token_indices = [init_token_idx]
            for _ in range(max_connections):
                if cur_token_indices[-1] in next_token_idx_dict:
                    if (
                        next_token_idx_dict[cur_token_indices[-1]]
                        not in init_token_indices
                    ):
                        cur_token_indices.append(
                            next_token_idx_dict[cur_token_indices[-1]]
                        )
                    else:
                        break
                else:
                    break
            sub_outputs.append(tuple(cur_token_indices))

        outputs.append(sub_outputs)

    return outputs


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
