"""Optional BERT wrapper without machine-specific paths."""

from typing import Optional

import torch
from torch import Tensor, nn


class BertTextEncoder(nn.Module):
    """Create 768-D contextual token features from BERT token tensors.

    ``pretrained_name_or_path`` can be the Hugging Face model id
    ``bert-base-uncased`` or a user-supplied local directory. Nothing is
    hardcoded to an author's filesystem.
    """

    def __init__(
        self,
        pretrained_name_or_path: str = "bert-base-uncased",
        fine_tune: bool = True,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "Install the optional 'transformers' dependency to encode raw text tokens"
            ) from exc
        self.model = AutoModel.from_pretrained(pretrained_name_or_path)
        self.fine_tune = fine_tune

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Optional[Tensor] = None,
    ) -> Tensor:
        kwargs = {
            "input_ids": input_ids.long(),
            "attention_mask": attention_mask.long(),
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids.long()
        if self.fine_tune:
            return self.model(**kwargs).last_hidden_state
        self.model.eval()
        with torch.no_grad():
            return self.model(**kwargs).last_hidden_state
