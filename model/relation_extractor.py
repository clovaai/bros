"""
BROS
Copyright 2022-present NAVER Corp.
Apache License v2.0
"""

import torch
from torch import nn


class RelationExtractor(nn.Module):
    def __init__(
        self,
        n_relations,
        backbone_hidden_size,
        head_hidden_size,
        head_p_dropout=0.1,
    ):
        super().__init__()

        self.n_relations = n_relations
        self.backbone_hidden_size = backbone_hidden_size
        self.head_hidden_size = head_hidden_size
        self.head_p_dropout = head_p_dropout

        self.drop = nn.Dropout(head_p_dropout)
        self.q_net = nn.Linear(
            self.backbone_hidden_size, self.n_relations * self.head_hidden_size
        )

        self.k_net = nn.Linear(
            self.backbone_hidden_size, self.n_relations * self.head_hidden_size
        )

        self.dummy_node = nn.Parameter(torch.Tensor(1, self.backbone_hidden_size))
        nn.init.normal_(self.dummy_node)

    def forward(self, h_q, h_k):
        h_q = self.q_net(self.drop(h_q))

        dummy_vec = self.dummy_node.unsqueeze(0).repeat(1, h_k.size(1), 1)
        h_k = torch.cat([h_k, dummy_vec], axis=0)
        h_k = self.k_net(self.drop(h_k))

        head_q = h_q.view(
            h_q.size(0), h_q.size(1), self.n_relations, self.head_hidden_size
        )
        head_k = h_k.view(
            h_k.size(0), h_k.size(1), self.n_relations, self.head_hidden_size
        )

        relation_score = torch.einsum("ibnd,jbnd->nbij", (head_q, head_k))

        return relation_score
