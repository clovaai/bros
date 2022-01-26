"""
BROS
Copyright 2022-present NAVER Corp.
Apache License v2.0
"""

import torch
from torch import nn
from transformers import LayoutLMConfig, LayoutLMModel, LayoutLMTokenizer

from bros import BrosConfig, BrosModel, BrosTokenizer
from model.relation_extractor import RelationExtractor


class BROSSPADERELModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.model_cfg = cfg.model
        if self.model_cfg.backbone in [
            "naver-clova-ocr/bros-base-uncased",
            "naver-clova-ocr/bros-large-uncased",
        ]:
            self.backbone_config = BrosConfig.from_pretrained(self.model_cfg.backbone)
            self.tokenizer = BrosTokenizer.from_pretrained(self.model_cfg.backbone)
            self.backbone = BrosModel.from_pretrained(self.model_cfg.backbone)
        elif self.model_cfg.backbone in [
            "microsoft/layoutlm-base-uncased",
            "microsoft/layoutlm-large-uncased",
        ]:
            self.backbone_config = LayoutLMConfig.from_pretrained(
                self.model_cfg.backbone
            )
            self.tokenizer = LayoutLMTokenizer.from_pretrained(self.model_cfg.backbone)
            self.backbone = LayoutLMModel.from_pretrained(self.model_cfg.backbone)
        else:
            raise ValueError(
                f"Not supported model: self.model_cfg.backbone={self.model_cfg.backbone}"
            )

        self._create_head()

        self.loss_func = nn.CrossEntropyLoss()

    def _create_head(self):
        self.backbone_hidden_size = self.backbone_config.hidden_size
        self.head_hidden_size = self.model_cfg.head_hidden_size
        self.head_p_dropout = self.model_cfg.head_p_dropout
        self.n_classes = self.model_cfg.n_classes + 1

        self.relation_net = RelationExtractor(
            n_relations=1,
            backbone_hidden_size=self.backbone_hidden_size,
            head_hidden_size=self.head_hidden_size,
            head_p_dropout=self.head_p_dropout,
        )
        self.relation_net.apply(self._init_weight)

    @staticmethod
    def _init_weight(module):
        init_std = 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.0, init_std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.normal_(module.weight, 1.0, init_std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, batch):
        input_ids = batch["input_ids"]
        bbox = batch["bbox"]
        attention_mask = batch["attention_mask"]

        backbone_outputs = self.backbone(
            input_ids=input_ids, bbox=bbox, attention_mask=attention_mask
        )
        last_hidden_states = backbone_outputs.last_hidden_state

        last_hidden_states = last_hidden_states.transpose(0, 1).contiguous()

        el_outputs = self.relation_net(last_hidden_states, last_hidden_states).squeeze(
            0
        )

        head_outputs = {"el_outputs": el_outputs}

        loss = self._get_loss(head_outputs, batch)

        return head_outputs, loss

    def _get_loss(self, head_outputs, batch):
        el_outputs = head_outputs["el_outputs"]

        bsz, max_seq_length = batch["attention_mask"].shape
        device = batch["attention_mask"].device

        self_token_mask = (
            torch.eye(max_seq_length, max_seq_length + 1).to(device).bool()
        )

        box_first_token_mask = torch.cat(
            [
                (batch["are_box_first_tokens"] == False),
                torch.zeros([bsz, 1], dtype=torch.bool).to(device),
            ],
            axis=1,
        )
        el_outputs.masked_fill_(box_first_token_mask[:, None, :], -10000.0)
        el_outputs.masked_fill_(self_token_mask[None, :, :], -10000.0)

        mask = batch["are_box_first_tokens"].view(-1)

        logits = el_outputs.view(-1, max_seq_length + 1)
        logits = logits[mask]

        labels = batch["el_labels"].view(-1)
        labels = labels[mask]

        loss = self.loss_func(logits, labels)

        return loss
