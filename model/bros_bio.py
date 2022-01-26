"""
BROS
Copyright 2022-present NAVER Corp.
Apache License v2.0
"""

from torch import nn
from transformers import LayoutLMConfig, LayoutLMModel, LayoutLMTokenizer

from bros import BrosConfig, BrosModel, BrosTokenizer


class BROSBIOModel(nn.Module):
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
        self.bio_layer = nn.Linear(
            self.backbone_config.hidden_size, self.model_cfg.n_classes
        )
        self.bio_layer.apply(self._init_weight)

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

        head_outputs = self.bio_layer(last_hidden_states)

        loss = self._get_loss(head_outputs, batch)

        return head_outputs, loss

    def _get_loss(self, head_outputs, batch):
        mask = batch["are_box_first_tokens"].view(-1)

        logits = head_outputs.view(-1, self.model_cfg.n_classes)
        logits = logits[mask]

        labels = batch["bio_labels"].view(-1)
        labels = labels[mask]

        loss = self.loss_func(logits, labels)

        return loss
