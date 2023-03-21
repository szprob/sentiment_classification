from typing import Dict, Optional

import torch
from torch import nn

from sentiment_classification.bert.attention import ScaledDotProductAttention
from sentiment_classification.bert.bert import BERT


class CoralLayer(torch.nn.Module):
    """Modified from coral layer.

    Implements CORAL layer described in
    Cao, Mirjalili, and Raschka (2020)
    *Rank Consistent Ordinal Regression for Neural Networks
       with Application to Age Estimation*
    Pattern Recognition Letters, https://doi.org/10.1016/j.patrec.2020.11.008
    Parameters
    -----------
    size_in : int
        Number of input features for the inputs to the forward method, which
        are expected to have shape=(num_examples, num_features).
    num_classes : int
        Number of classes in the dataset.
    preinit_bias : bool (default=True)
        If true, it will pre-initialize the biases to descending values in
        [0, 1] range instead of initializing it to all zeros. This pre-
        initialization scheme results in faster learning and better
        generalization performance in practice.
    """

    def __init__(self, size_in, num_classes, preinit_bias=True):
        super().__init__()
        self.size_in, self.size_out = size_in, 1

        self.coral_weights = torch.nn.Linear(self.size_in, 1, bias=False)
        if preinit_bias:
            self.coral_bias = torch.nn.Parameter(
                torch.arange(num_classes, 0, -1).float() / num_classes
            )
        else:
            self.coral_bias = torch.nn.Parameter(torch.zeros(num_classes).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes forward pass.
        Parameters
        -----------
        x : torch.tensor, shape=(num_examples, num_features)
            Input features.
        Returns
        -----------
        logits : torch.tensor, shape=(num_examples, num_classes-1)
        """
        return self.coral_weights(x) + self.coral_bias


class Head(nn.Module):
    """cls head for sentiment classfication."""

    def __init__(self, hidden_size: int = 512, tag_nums: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.tag_nums = tag_nums
        self.q = nn.Parameter(torch.rand(1, 1, hidden_size))
        self.att = ScaledDotProductAttention()
        self.cls = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(negative_slope=0.1),
            CoralLayer(self.hidden_size, self.tag_nums),
            # nn.Sigmoid(),
        )

    def forward(self, seqs: torch.Tensor) -> torch.Tensor:
        batch_size = seqs.shape[0]
        # q (b ,1 ,d)
        q = self.q.expand(batch_size, 1, self.hidden_size)
        # x (b,1,d)
        x = self.att(q, seqs, seqs)
        # x (b,d)
        x = x.squeeze(1)
        out = self.cls(x)
        return out


class Model(nn.Module):
    """sentiment classification.

    Given a encoded text,`Model` will get a sentiment socre on it.

    Attributes:
        config (Optional[Dict],optional):
            Modeling config.
            Defaults to None.
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        self.config = {} if config is None else config
        self.bert = BERT(config=self.config)

        self.head2 = Head(
            hidden_size=self.config.get("hidden_size", 512),
            tag_nums=self.config.get("head2_tag_num", 1),
        )
        self.head3 = Head(
            hidden_size=self.config.get("hidden_size", 512),
            tag_nums=self.config.get("head3_tag_num", 2),
        )
        self.head5 = Head(
            hidden_size=self.config.get("hidden_size", 512),
            tag_nums=self.config.get("head5_tag_num", 4),
        )

    def forward2(self, x: torch.Tensor) -> torch.Tensor:
        seqs = self.bert(x)
        out2 = self.head2(seqs)
        return out2

    def forward3(self, x: torch.Tensor) -> torch.Tensor:
        seqs = self.bert(x)
        out3 = self.head3(seqs)
        return out3

    def forward5(self, x: torch.Tensor) -> torch.Tensor:
        seqs = self.bert(x)
        out5 = self.head5(seqs)
        return out5

    def forward(
        self, x2: torch.Tensor, x3: torch.Tensor, x5: torch.Tensor
    ) -> torch.Tensor:
        return self.forward2(x2), self.forward3(x3), self.forward5(x5)

    @torch.no_grad()
    def score(self, input: torch.Tensor) -> torch.Tensor:
        """Scoring the input text(one input).

        Args:
            input (torch.Tensor):
                Text input(should be encoded by bert tokenizer.)

        Returns:
            torch.Tensor:
                The toxic score of the input .
        """

        return torch.sigmoid(self.forward5(input)).detach().cpu()
