"""
BROS
Copyright 2022-present NAVER Corp.
Apache License v2.0

Example:
    python evaluate.py \
        --config=configs/finetune_funsd_ee_bies.yaml \
        --pretrained_model_file=finetune_funsd_ee_bies__bros-base-uncased/checkpoints/epoch=99-last.pt
"""


import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from lightning_modules.bros_bies_module import get_label_map
from lightning_modules.data_modules.bros_dataset import BROSDataset
from model import get_model
from utils import get_class_names, get_config


def main():
    cfg = get_config()
    print(cfg)

    net = get_model(cfg)

    load_model_weight(net, cfg.pretrained_model_file)

    net.to("cuda")
    net.eval()

    if cfg.model.backbone in [
        "naver-clova-ocr/bros-base-uncased",
        "naver-clova-ocr/bros-large-uncased",
    ]:
        backbone_type = "bros"
    elif cfg.model.backbone in [
        "microsoft/layoutlm-base-uncased",
        "microsoft/layoutlm-large-uncased",
    ]:
        backbone_type = "layoutlm"
    else:
        raise ValueError(
            f"Not supported model: self.cfg.model.backbone={cfg.model.backbone}"
        )

    mode = "val"

    dataset = BROSDataset(
        cfg.dataset,
        cfg.task,
        backbone_type,
        cfg.model.head,
        cfg.dataset_root_path,
        net.tokenizer,
        mode=mode,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=cfg[mode].batch_size,
        shuffle=False,
        num_workers=cfg[mode].num_workers,
        pin_memory=True,
        drop_last=False,
    )

    if cfg.model.head == "bies":
        from lightning_modules.bros_bies_module import do_eval_epoch_end, do_eval_step

        eval_kwargs = get_eval_kwargs_bies(cfg.dataset_root_path)
    elif cfg.model.head == "bio":
        from lightning_modules.bros_bio_module import do_eval_epoch_end, do_eval_step

        eval_kwargs = get_eval_kwargs_bio(cfg.dataset_root_path)
    elif cfg.model.head == "spade":
        from lightning_modules.bros_spade_module import do_eval_epoch_end, do_eval_step

        eval_kwargs = get_eval_kwargs_spade(
            cfg.dataset_root_path, cfg.train.max_seq_length
        )
    elif cfg.model.head == "spade_rel":
        from lightning_modules.bros_spade_rel_module import (
            do_eval_epoch_end,
            do_eval_step,
        )

        eval_kwargs = get_eval_kwargs_spade_rel(cfg.train.max_seq_length)
    else:
        raise ValueError(f"Unknown cfg.config={cfg.config}")

    step_outputs = []
    for example_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        # Convert batch tensors to given device
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(net.backbone.device)

        with torch.no_grad():
            head_outputs, loss = net(batch)
        step_out = do_eval_step(batch, head_outputs, loss, eval_kwargs)
        step_outputs.append(step_out)

    # Get scores
    scores = do_eval_epoch_end(step_outputs)
    print(
        f"precision: {scores['precision']:.4f}, "
        f"recall: {scores['recall']:.4f}, "
        f"f1: {scores['f1']:.4f}"
    )


def load_model_weight(net, pretrained_model_file):
    pretrained_model_state_dict = torch.load(pretrained_model_file, map_location="cpu")[
        "state_dict"
    ]
    new_state_dict = {}
    for k, v in pretrained_model_state_dict.items():
        new_k = k
        if new_k.startswith("net."):
            new_k = new_k[len("net.") :]
        new_state_dict[new_k] = v
    net.load_state_dict(new_state_dict)


def get_eval_kwargs_bies(dataset_root_path):
    ignore_index = -100
    label_map = get_label_map(dataset_root_path)

    eval_kwargs = {
        "ignore_index": ignore_index,
        "label_map": label_map,
    }

    return eval_kwargs


def get_eval_kwargs_bio(dataset_root_path):
    class_names = get_class_names(dataset_root_path)

    eval_kwargs = {"class_names": class_names}

    return eval_kwargs


def get_eval_kwargs_spade(dataset_root_path, max_seq_length):
    class_names = get_class_names(dataset_root_path)
    dummy_idx = max_seq_length

    eval_kwargs = {"class_names": class_names, "dummy_idx": dummy_idx}

    return eval_kwargs


def get_eval_kwargs_spade_rel(max_seq_length):
    dummy_idx = max_seq_length

    eval_kwargs = {"dummy_idx": dummy_idx}

    return eval_kwargs


if __name__ == "__main__":
    main()
