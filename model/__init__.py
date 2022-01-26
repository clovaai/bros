"""
BROS
Copyright 2022-present NAVER Corp.
Apache License v2.0
"""

from model.bros_bies import BROSBIESModel
from model.bros_bio import BROSBIOModel
from model.bros_spade import BROSSPADEModel
from model.bros_spade_rel import BROSSPADERELModel


def get_model(cfg):
    if cfg.model.head == "bies":
        model = BROSBIESModel(cfg=cfg)
    elif cfg.model.head == "bio":
        model = BROSBIOModel(cfg=cfg)
    elif cfg.model.head == "spade":
        model = BROSSPADEModel(cfg=cfg)
    elif cfg.model.head == "spade_rel":
        model = BROSSPADERELModel(cfg=cfg)
    else:
        raise ValueError(f"Unknown cfg.model.head={cfg.model.head}")

    return model
