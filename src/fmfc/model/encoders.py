"""Injectable tiny encoders and local-only frozen pretrained towers."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor


class EncoderProtocol(Protocol):
    output_dim: int
    fingerprint: str

    def encode(self, texts: Sequence[str]) -> Tensor: ...


class TinyTextEncoder:
    """Deterministic byte-count features for synthetic mechanics."""

    def __init__(self, output_dim: int = 16) -> None:
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        self.output_dim = output_dim
        self.fingerprint = f"synthetic-byte-text-v1:{output_dim}"

    def encode(self, texts: Sequence[str]) -> Tensor:
        result = torch.zeros((len(texts), self.output_dim))
        for row, text in enumerate(texts):
            for value in text.lower().encode("utf-8"):
                result[row, value % self.output_dim] += 1.0
        return torch.nn.functional.normalize(result, dim=-1)


class TinyVisionEncoder:
    """Fixed image-tensor pooling for synthetic tensors; does not read media files."""

    def __init__(self, output_dim: int = 12) -> None:
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        self.output_dim = output_dim
        self.fingerprint = f"synthetic-pool-vision-v1:{output_dim}"

    def encode(self, pixels: Tensor) -> Tensor:
        if pixels.ndim != 4:
            raise ValueError("Expected image tensors [N,C,H,W]")
        if pixels.shape[0] == 0:
            return pixels.new_zeros((0, self.output_dim))
        return torch.nn.functional.adaptive_avg_pool1d(
            pixels.flatten(1).unsqueeze(1), self.output_dim
        ).squeeze(1)

    def encode_paths(self, paths: Sequence[str]) -> Tensor:
        if paths:
            raise ValueError("TinyVisionEncoder accepts synthetic tensors, not benchmark media")
        return torch.zeros((0, self.output_dim))


class FrozenTextEncoder:
    def __init__(
        self, checkpoint_path: str | Path, revision: str | None = None,
        max_length: int = 256, device: str = "cpu",
    ) -> None:
        root = str(checkpoint_path)
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(root, revision=revision, local_files_only=True)
        self.model = AutoModel.from_pretrained(root, revision=revision, local_files_only=True).to(device)
        self.model.requires_grad_(False).eval()
        self.output_dim = self.model.config.hidden_size
        self.device = device
        self.max_length = max_length
        self.truncated_items = 0
        self.fingerprint = f"text:{root}:{revision}:mean-pool:{max_length}"

    @torch.no_grad()
    def encode(self, texts: Sequence[str]) -> Tensor:
        if not texts:
            return torch.zeros((0, self.output_dim))
        lengths = self.tokenizer(
            list(texts), truncation=False, add_special_tokens=True
        )["input_ids"]
        self.truncated_items += sum(len(tokens) > self.max_length for tokens in lengths)
        tokens = self.tokenizer(
            list(texts), padding=True, truncation=True, max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        output = self.model(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1)
        pooled = (output * mask).sum(1) / mask.sum(1).clamp_min(1)
        return pooled.detach().cpu().clone()


class FrozenTokenTextEncoder(FrozenTextEncoder):
    """Frozen sequence states for the unchanged CrossTrans attention blocks."""

    def __init__(
        self, checkpoint_path: str | Path, revision: str | None = None,
        max_length: int = 256, device: str = "cpu",
    ) -> None:
        super().__init__(checkpoint_path, revision, max_length, device)
        self.fingerprint = f"text:{checkpoint_path}:{revision}:token-states:{max_length}"

    @torch.no_grad()
    def encode_tokens(self, texts: Sequence[str]) -> tuple[Tensor, Tensor]:
        tokens = self.tokenizer(
            list(texts), padding=True, truncation=True, max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        output = self.model(**tokens).last_hidden_state
        return (
            output.detach().cpu().clone().float(),
            tokens["attention_mask"].detach().cpu().clone().bool(),
        )


class FrozenVisionEncoder:
    def __init__(
        self, checkpoint_path: str | Path, revision: str | None = None,
        device: str = "cpu",
    ) -> None:
        root = str(checkpoint_path)
        from transformers import AutoImageProcessor, CLIPVisionModel

        self.processor = AutoImageProcessor.from_pretrained(root, revision=revision, local_files_only=True)
        self.model = CLIPVisionModel.from_pretrained(root, revision=revision, local_files_only=True).to(device)
        self.model.requires_grad_(False).eval()
        self.output_dim = self.model.config.hidden_size
        self.device = device
        self.fingerprint = f"vision:{root}:{revision}:clip-pool-v1"

    @torch.no_grad()
    def encode_paths(self, paths: Sequence[str]) -> Tensor:
        if not paths:
            return torch.zeros((0, self.output_dim))
        from PIL import Image

        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        return self.model(**inputs).pooler_output.detach().cpu().clone()


class FrozenCLIPImageEncoder:
    """Normalized projected CLIP features, as used by CrossTransVFC's image path."""

    def __init__(
        self, checkpoint_path: str | Path, revision: str | None = None,
        device: str = "cpu",
    ) -> None:
        from transformers import AutoImageProcessor, CLIPVisionModelWithProjection

        root = str(checkpoint_path)
        self.processor = AutoImageProcessor.from_pretrained(
            root, revision=revision, local_files_only=True,
        )
        self.model = CLIPVisionModelWithProjection.from_pretrained(
            root, revision=revision, local_files_only=True,
        ).to(device)
        self.model.requires_grad_(False).eval()
        self.output_dim = self.model.config.projection_dim
        self.device = device
        self.fingerprint = f"vision:{root}:{revision}:clip-normalized-projection"

    @torch.no_grad()
    def encode_paths(self, paths: Sequence[str]) -> Tensor:
        if not paths:
            return torch.zeros((0, self.output_dim))
        from PIL import Image

        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        features = self.model(**inputs).image_embeds
        return torch.nn.functional.normalize(features, dim=-1).detach().cpu().clone()
