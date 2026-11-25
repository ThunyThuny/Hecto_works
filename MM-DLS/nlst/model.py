"""MM-DLS whole-CT transfer for confirmed lung-cancer prognosis.

New NLST class convention: subtype 0=LUAD, 1=LUSC; stage 0=I-II, 1=III,
2=IV. These heads are freshly initialized; the original checkpoint's head
label ordering and training provenance are not assumed to match NLST.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from mm_dls.LesionAttentionFusion import LesionAttentionFusion
from mm_dls.ModelSpaceEncoder import SpaceEncoder


class NLSTPrognosisModel(nn.Module):
    """Single whole-CT stream plus real clinical features and missing indicators.

    Only the original space_encoder and space_fuser are transferred. There is
    no invented lesion stream, PET or radiomics input. The fresh hierarchical
    classifier/survival layers are supervised by available labels only. A DFS
    head is absent by default; it may only be enabled after genuine disease-free
    survival labels have been verified by the dataset/trainer.
    """

    TRANSFER_PREFIXES = ("space_encoder.", "space_fuser.")
    SUBTYPE_CLASSES = ("LUAD", "LUSC")
    STAGE_CLASSES = ("I-II", "III", "IV")
    HORIZON_MONTHS = (12, 36, 60)

    def __init__(
        self,
        clinical_dim: int = 12,
        enabled_tasks: tuple[str, ...] = ("subtype", "stage", "os"),
        img_feat_dim: int = 128,
        hidden_dim: int = 128,
        task_emb_dim: int = 32,
        dropout: float = 0.3,
        encode_chunk_size: int = 32,
    ) -> None:
        super().__init__()
        if not isinstance(clinical_dim, int) or clinical_dim < 0:
            raise ValueError("clinical_dim must be a nonnegative integer")
        for name, value in (
            ("img_feat_dim", img_feat_dim),
            ("hidden_dim", hidden_dim),
            ("task_emb_dim", task_emb_dim),
            ("encode_chunk_size", encode_chunk_size),
        ):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if img_feat_dim % 4:
            raise ValueError("img_feat_dim must be divisible by four attention heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")

        if not enabled_tasks or isinstance(enabled_tasks, str):
            raise ValueError("enabled_tasks must be a nonempty sequence of task names")
        allowed_tasks = {"subtype", "stage", "os", "dfs"}
        if len(set(enabled_tasks)) != len(enabled_tasks) or not set(enabled_tasks) <= allowed_tasks:
            raise ValueError("enabled_tasks must be unique names from subtype, stage, os, dfs")
        self.enabled_tasks = tuple(enabled_tasks)
        self.clinical_dim = clinical_dim
        self.img_feat_dim = img_feat_dim
        self.hidden_dim = hidden_dim
        self.task_emb_dim = task_emb_dim
        self.encode_chunk_size = encode_chunk_size
        self.space_encoder = SpaceEncoder(input_channels=1, feature_dim=img_feat_dim)
        self.space_fuser = LesionAttentionFusion(img_feat_dim, img_feat_dim)
        self.fusion = nn.Sequential(
            nn.Linear(img_feat_dim + clinical_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        self.subtype_head = (
            nn.Linear(hidden_dim, len(self.SUBTYPE_CLASSES)) if "subtype" in self.enabled_tasks else None
        )
        self.stage_head = (
            nn.Linear(hidden_dim, len(self.STAGE_CLASSES)) if "stage" in self.enabled_tasks else None
        )
        has_survival = bool({"os", "dfs"}.intersection(self.enabled_tasks))
        self.subtype_emb = (
            nn.Embedding(len(self.SUBTYPE_CLASSES), task_emb_dim)
            if has_survival and self.subtype_head is not None else None
        )
        self.stage_emb = (
            nn.Embedding(len(self.STAGE_CLASSES), task_emb_dim)
            if has_survival and self.stage_head is not None else None
        )
        condition_count = int(self.subtype_emb is not None) + int(self.stage_emb is not None)
        self.survival_mlp = (
            nn.Sequential(
                nn.Linear(hidden_dim + condition_count * task_emb_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ) if has_survival else None
        )
        self.os_head = nn.Linear(hidden_dim, 1) if "os" in self.enabled_tasks else None
        self.os_cls = (
            nn.Linear(hidden_dim, len(self.HORIZON_MONTHS)) if "os" in self.enabled_tasks else None
        )
        self.dfs_head = nn.Linear(hidden_dim, 1) if "dfs" in self.enabled_tasks else None
        self.dfs_cls = (
            nn.Linear(hidden_dim, len(self.HORIZON_MONTHS)) if "dfs" in self.enabled_tasks else None
        )
        self._backbone_frozen = False
        self.initialization_metadata: dict[str, Any] = {"source": "random"}

    def config(self) -> dict[str, Any]:
        """Constructor arguments sufficient to recreate a saved model."""
        return {
            "clinical_dim": self.clinical_dim,
            "enabled_tasks": list(self.enabled_tasks),
            "img_feat_dim": self.img_feat_dim,
            "hidden_dim": self.hidden_dim,
            "task_emb_dim": self.task_emb_dim,
            "dropout": self.fusion[2].p,
            "encode_chunk_size": self.encode_chunk_size,
        }

    def backbone_parameters(self):
        yield from self.space_encoder.parameters()
        yield from self.space_fuser.parameters()

    def head_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith(self.TRANSFER_PREFIXES):
                yield parameter

    def freeze_backbone(self, frozen: bool = True) -> None:
        """Freeze during head warmup; pass False for subsequent fine-tuning."""
        self._backbone_frozen = bool(frozen)
        for parameter in self.backbone_parameters():
            parameter.requires_grad_(not self._backbone_frozen)
        self.space_encoder.train(self.training and not self._backbone_frozen)
        self.space_fuser.train(self.training and not self._backbone_frozen)

    def train(self, mode: bool = True):
        super().train(mode)
        if self._backbone_frozen:
            self.space_encoder.eval()
            self.space_fuser.eval()
        return self

    def _encode_volume(self, ct: Tensor) -> Tensor:
        if ct.ndim != 5 or ct.shape[2] != 1:
            raise ValueError("ct must have shape [batch,slices,1,height,width]")
        if any(size < 1 for size in ct.shape):
            raise ValueError("ct dimensions must be nonzero")
        if not ct.is_floating_point():
            raise ValueError("ct must be a floating-point tensor")
        batch, slices, channels, height, width = ct.shape
        flat = ct.reshape(batch * slices, channels, height, width)
        features = torch.cat(
            [self.space_encoder(chunk) for chunk in flat.split(self.encode_chunk_size)],
            dim=0,
        )
        return self.space_fuser(features.reshape(batch, slices, self.img_feat_dim))

    def forward(self, ct: Tensor, clinical: Tensor | None = None) -> dict[str, Tensor]:
        image_features = self._encode_volume(ct)
        if self.clinical_dim:
            expected_shape = (ct.shape[0], self.clinical_dim)
            if clinical is None or tuple(clinical.shape) != expected_shape:
                raise ValueError(f"clinical must have shape {expected_shape}")
            clinical = clinical.to(device=image_features.device, dtype=image_features.dtype)
            base = torch.cat([image_features, clinical], dim=1)
        else:
            if clinical is not None and tuple(clinical.shape) != (ct.shape[0], 0):
                raise ValueError("clinical_dim=0 only accepts None or an empty [batch,0] tensor")
            base = image_features
        shared = self.fusion(base)
        outputs = {}
        conditions = [shared]
        if self.subtype_head is not None:
            outputs["subtype_logits"] = self.subtype_head(shared)
            if self.subtype_emb is not None:
                conditions.append(torch.softmax(outputs["subtype_logits"], dim=1) @ self.subtype_emb.weight)
        if self.stage_head is not None:
            outputs["stage_logits"] = self.stage_head(shared)
            if self.stage_emb is not None:
                conditions.append(torch.softmax(outputs["stage_logits"], dim=1) @ self.stage_emb.weight)
        if self.survival_mlp is not None:
            survival = self.survival_mlp(torch.cat(conditions, dim=1))
            if self.os_head is not None:
                outputs["os_risk"] = self.os_head(survival).squeeze(1)
                outputs["os_logits"] = self.os_cls(survival)
            if self.dfs_head is not None:
                outputs["dfs_risk"] = self.dfs_head(survival).squeeze(1)
                outputs["dfs_logits"] = self.dfs_cls(survival)
        return outputs

    def load_pretrained(self, checkpoint_path: str | Path) -> dict[str, Any]:
        """Load only verified whole-CT parameters, preserving all fresh heads.

        The restricted weights-only loader accepts a plain state dictionary or
        a dictionary wrapped under model_state_dict, state_dict or model. Any
        missing/incompatible required CT tensor fails before model mutation.
        """
        checkpoint_path = Path(checkpoint_path)
        source_hash = hashlib.sha256()
        with checkpoint_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                source_hash.update(chunk)
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(raw, Mapping):
            raise ValueError("checkpoint must be a state dict or contain a state dict")
        state = raw
        for wrapper in ("model_state_dict", "state_dict", "model"):
            candidate = raw.get(wrapper)
            if isinstance(candidate, Mapping):
                state = candidate
                break

        target = self.state_dict()
        selected: dict[str, Tensor] = {}
        dropped: list[dict[str, Any]] = []
        for raw_key, value in state.items():
            key = str(raw_key)
            if key.startswith("module."):
                key = key[len("module.") :]
            if not key.startswith(self.TRANSFER_PREFIXES):
                dropped.append({"key": str(raw_key), "reason": "outside whole-CT transfer whitelist"})
                continue
            if key not in target:
                dropped.append({"key": str(raw_key), "reason": "unknown target key"})
                continue
            if not isinstance(value, Tensor):
                dropped.append({"key": str(raw_key), "reason": "not a tensor"})
                continue
            if tuple(value.shape) != tuple(target[key].shape):
                dropped.append(
                    {
                        "key": str(raw_key),
                        "reason": "shape mismatch",
                        "source_shape": list(value.shape),
                        "target_shape": list(target[key].shape),
                    }
                )
                continue
            if value.dtype != target[key].dtype:
                raise ValueError(f"incompatible checkpoint dtype for {raw_key}: {value.dtype}")
            if not torch.is_floating_point(value) or not torch.isfinite(value).all().item():
                raise ValueError(f"non-floating or non-finite backbone weight: {raw_key}")
            if key in selected:
                raise ValueError(f"duplicate normalized checkpoint key: {key}")
            selected[key] = value

        expected = [key for key in target if key.startswith(self.TRANSFER_PREFIXES)]
        missing = sorted(set(expected).difference(selected))
        if not any(key.startswith("space_encoder.") for key in selected):
            raise ValueError("checkpoint contains no compatible whole-CT encoder weights")
        if missing:
            raise ValueError("missing compatible whole-CT backbone weights: " + ", ".join(missing))

        tensor_hashes = {}
        transfer_hash = hashlib.sha256()
        for key in sorted(selected):
            value = selected[key].detach().cpu().contiguous()
            payload = value.view(torch.uint8).numpy().tobytes()
            tensor_hashes[key] = hashlib.sha256(payload).hexdigest()
            transfer_hash.update(key.encode("utf-8"))
            transfer_hash.update(str(value.dtype).encode("ascii"))
            transfer_hash.update(str(tuple(value.shape)).encode("ascii"))
            transfer_hash.update(payload)

        merged_state = dict(target)
        merged_state.update(selected)
        self.load_state_dict(merged_state, strict=True)
        metadata = {
            "source": str(checkpoint_path.resolve()),
            "checkpoint_sha256": source_hash.hexdigest(),
            "transferred_state_sha256": transfer_hash.hexdigest(),
            "loaded_tensor_sha256": tensor_hashes,
            "loaded_keys": sorted(selected),
            "loaded_key_count": len(selected),
            "loaded_parameter_count": sum(value.numel() for value in selected.values()),
            "dropped_key_count": len(dropped),
            "dropped": dropped,
            "transferred_components": ["space_encoder", "space_fuser"],
            "fresh_components": [
                name for name, _ in self.named_children()
                if name not in ("space_encoder", "space_fuser")
            ],
            "enabled_tasks": list(self.enabled_tasks),
            "subtype_classes": list(self.SUBTYPE_CLASSES),
            "stage_classes": list(self.STAGE_CLASSES),
            "horizon_months": list(self.HORIZON_MONTHS),
            "dfs_training_requires_verified_endpoint": True,
        }
        self.initialization_metadata = metadata
        return metadata
