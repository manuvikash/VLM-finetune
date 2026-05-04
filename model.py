#!/usr/bin/env python3
"""
CLIP ViT-B/32 backbone + classification heads for flat / hierarchy-aware training.
"""

from __future__ import annotations

from typing import Literal

import open_clip
import torch
import torch.nn as nn


def create_clip_preprocesses():
    """Returns (train_preprocess, val_preprocess) for ViT-B-32."""
    _, train_pp, val_pp = open_clip.create_model_and_transforms(
        model_name="ViT-B-32",
        pretrained="openai",
    )
    return train_pp, val_pp


def _infer_visual_embed_dim(visual: nn.Module, *, device: torch.device) -> int:
    visual.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224, device=device, dtype=torch.float32)
        out = visual(dummy)
    return int(out.shape[-1])


class CLIPHierClassifier(nn.Module):
    """
    Image encoder: OpenCLIP ViT-B/32 visual tower.
    Optional parent head when mode=='hierarchy'.
    """

    def __init__(
        self,
        num_leaf: int,
        *,
        mode: Literal["flat", "hierarchy"] = "flat",
        num_parent: int | None = None,
        pretrained: str = "openai",
    ) -> None:
        super().__init__()
        self.mode = mode

        full, _, _ = open_clip.create_model_and_transforms(
            model_name="ViT-B-32",
            pretrained=pretrained,
            device="cpu",
            jit=False,
        )

        self.visual = full.visual
        if mode == "hierarchy":
            if num_parent is None or num_parent < 2:
                raise ValueError("hierarchy mode requires num_parent >= 2")
            self.num_parent = num_parent

        infer_dev = torch.device("cpu")
        feat_dim = _infer_visual_embed_dim(self.visual, device=infer_dev)

        self.leaf_head = nn.Linear(feat_dim, num_leaf)
        self.parent_head: nn.Linear | None
        if mode == "hierarchy":
            assert num_parent is not None
            self.parent_head = nn.Linear(feat_dim, num_parent)
        else:
            self.parent_head = None

        nn.init.normal_(self.leaf_head.weight, std=0.02)
        nn.init.zeros_(self.leaf_head.bias)
        if self.parent_head is not None:
            nn.init.normal_(self.parent_head.weight, std=0.02)
            nn.init.zeros_(self.parent_head.bias)

    def freeze_backbone(self, freeze: bool) -> None:
        """If True, disables gradients on visual tower weights."""
        for p in self.visual.parameters():
            p.requires_grad = not freeze

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.visual(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        z = self.forward_features(x)
        leaf_logits = self.leaf_head(z)
        parent_logits = self.parent_head(z) if self.parent_head is not None else None
        return leaf_logits, parent_logits


@torch.no_grad()
def build_zeroshot_text_features(
    clip_model,
    tokenizer,
    leaf_names_ordered: list[str],
    device: torch.device,
    *,
    prompt_template: str = "a photo of a {}",
) -> torch.Tensor:
    """L2-normalized text embeddings aligned with indices in leaf_names_ordered."""
    prompts = [prompt_template.format(n.replace("_", " ")) for n in leaf_names_ordered]
    tokens = tokenizer(prompts).to(device)
    text_feats = clip_model.encode_text(tokens).float()
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    return text_feats
