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


class BROSSPADEModel(nn.Module):
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

        # (1) Initial token classification
        self.itc_layer = nn.Sequential(
            nn.Dropout(self.head_p_dropout),
            nn.Linear(self.backbone_hidden_size, self.backbone_hidden_size),
            nn.Dropout(self.head_p_dropout),
            nn.Linear(self.backbone_hidden_size, self.n_classes),
        )
        # (2) Subsequent token classification
        self.stc_layer = RelationExtractor(
            n_relations=1,
            backbone_hidden_size=self.backbone_hidden_size,
            head_hidden_size=self.head_hidden_size,
            head_p_dropout=self.head_p_dropout,
        )

        self.itc_layer.apply(self._init_weight)
        self.stc_layer.apply(self._init_weight)

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

        itc_outputs = self.itc_layer(last_hidden_states).transpose(0, 1).contiguous()
        stc_outputs = self.stc_layer(last_hidden_states, last_hidden_states).squeeze(0)
        head_outputs = {"itc_outputs": itc_outputs, "stc_outputs": stc_outputs}

        loss = self._get_loss(head_outputs, batch)

        return head_outputs, loss

    def _get_loss(self, head_outputs, batch):
        itc_outputs = head_outputs["itc_outputs"]
        stc_outputs = head_outputs["stc_outputs"]

        itc_loss = self._get_itc_loss(itc_outputs, batch)
        stc_loss = self._get_stc_loss(stc_outputs, batch)

        loss = itc_loss + stc_loss

        return loss

    def _get_itc_loss(self, itc_outputs, batch):
        itc_mask = batch["are_box_first_tokens"].view(-1)

        itc_logits = itc_outputs.view(-1, self.model_cfg.n_classes + 1)
        itc_logits = itc_logits[itc_mask]

        itc_labels = batch["itc_labels"].view(-1)
        itc_labels = itc_labels[itc_mask]

        itc_loss = self.loss_func(itc_logits, itc_labels)

        return itc_loss

    def _get_stc_loss(self, stc_outputs, batch):
        inv_attention_mask = 1 - batch["attention_mask"]

        bsz, max_seq_length = inv_attention_mask.shape
        device = inv_attention_mask.device

        invalid_token_mask = torch.cat(
            [inv_attention_mask, torch.zeros([bsz, 1]).to(device)], axis=1
        ).bool()
        stc_outputs.masked_fill_(invalid_token_mask[:, None, :], -10000.0)

        self_token_mask = (
            torch.eye(max_seq_length, max_seq_length + 1).to(device).bool()
        )
        stc_outputs.masked_fill_(self_token_mask[None, :, :], -10000.0)

        stc_mask = batch["attention_mask"].view(-1).bool()

        stc_logits = stc_outputs.view(-1, max_seq_length + 1)
        stc_logits = stc_logits[stc_mask]

        stc_labels = batch["stc_labels"].view(-1)
        stc_labels = stc_labels[stc_mask]

        stc_loss = self.loss_func(stc_logits, stc_labels)

        return stc_loss
