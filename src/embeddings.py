"""
Foundation model encoder wrappers with unified interface.

Supports UNI, Virchow2, and H-Optimus-0 with multi-GPU inference.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import timm
from huggingface_hub import hf_hub_download


class Encoder(ABC):
    """Base class for all encoders."""

    def __init__(self, device: str = "cuda", use_mixed_precision: bool = True):
        """
        Args:
            device: Device to run inference on
            use_mixed_precision: Whether to use automatic mixed precision
        """
        self.device = torch.device(device)
        self.use_mixed_precision = use_mixed_precision
        self.model = None
        self.transform = None

    @abstractmethod
    def _load_model(self):
        """Load the model architecture and weights."""
        pass

    @abstractmethod
    def _get_transform(self) -> transforms.Compose:
        """Get preprocessing transforms for this encoder."""
        pass

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """
        Preprocess images for the model.

        Args:
            images: Batch of images (B, 3, H, W) in float32 [0, 1]

        Returns:
            Preprocessed images (B, 3, H, W) in float32, normalized
        """
        images = images.float()

        # Apply model-specific normalization (ImageNet or custom)
        if self.transform is not None:
            images = torch.stack([self.transform(img) for img in images])

        return images

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> np.ndarray:
        """
        Extract embeddings from a batch of images.

        Args:
            images: Batch of images (B, 3, H, W) in float32 [0, 1]

        Returns:
            Embeddings (B, embed_dim) in float32
        """
        if self.model is None:
            self._load_model()
            self.model = self.model.to(self.device)
            self.model.eval()

        # Preprocess
        images = self.preprocess(images).to(self.device)

        # Forward pass with mixed precision
        if self.use_mixed_precision:
            with torch.cuda.amp.autocast():
                embeddings = self.model(images)
        else:
            embeddings = self.model(images)

        # Convert to numpy float32
        return embeddings.cpu().numpy().astype(np.float32)

    def __call__(self, images: torch.Tensor) -> np.ndarray:
        """Alias for encode()."""
        return self.encode(images)


class UNIEncoder(Encoder):
    """
    UNI (Universal Pathology Model) encoder.
    Vision Transformer trained on 100M+ histopathology images.
    """

    def __init__(self, model_id: str = "MahmoodLab/UNI", device: str = "cuda",
                 use_mixed_precision: bool = True):
        super().__init__(device, use_mixed_precision)
        self.model_id = model_id
        # UNI2-h uses ViT-Huge (1536 dim), original UNI uses ViT-Large (1024 dim)
        self.embed_dim = 1536 if "UNI2" in model_id else 1024
        self.transform = self._get_transform()
        self._load_model()
        # Move model to device and set to eval mode
        if self.model is not None:
            self.model = self.model.to(self.device)
            self.model.eval()

    def _load_model(self):
        """Load UNI model from HuggingFace."""
        try:
            if "UNI2" in self.model_id:
                # Use official UNI2-h configuration from HuggingFace
                # https://huggingface.co/MahmoodLab/UNI2-h
                timm_kwargs = {
                    'model_name': 'vit_giant_patch14_224',
                    'img_size': 224,
                    'patch_size': 14,
                    'depth': 24,
                    'num_heads': 24,
                    'init_values': 1e-5,  # LayerScale init
                    'embed_dim': 1536,
                    'mlp_ratio': 2.66667 * 2,  # 5.33333 → 8192 hidden dim
                    'num_classes': 0,
                    'no_embed_class': True,  # No CLS token in pos_embed
                    'mlp_layer': timm.layers.SwiGLUPacked,  # Built-in SwiGLU
                    'act_layer': nn.SiLU,
                    'reg_tokens': 8,  # Register tokens
                    'dynamic_img_size': True
                }
                self.model = timm.create_model(pretrained=False, **timm_kwargs)
            else:
                # Original UNI uses ViT-Large with patch16
                self.model = timm.create_model(
                    "vit_large_patch16_224",
                    pretrained=False,
                    num_classes=0,
                )

            # Load pretrained weights from HuggingFace
            checkpoint_path = hf_hub_download(
                repo_id=self.model_id,
                filename="pytorch_model.bin",
                cache_dir=".cache"
            )
            state_dict = torch.load(checkpoint_path, map_location="cpu")

            # Load weights (strict=True for UNI2-h as all keys should match now)
            if "UNI2" in self.model_id:
                self.model.load_state_dict(state_dict, strict=True)
                print(f"Info: Loaded UNI2-h checkpoint with official timm configuration")
            else:
                self.model.load_state_dict(state_dict, strict=False)

        except Exception as e:
            print(f"Warning: Could not load UNI from HuggingFace: {e}")
            if "UNI2" in self.model_id:
                print("Using random initialization with official UNI2-h config for testing")
                timm_kwargs = {
                    'model_name': 'vit_giant_patch14_224',
                    'img_size': 224,
                    'patch_size': 14,
                    'depth': 24,
                    'num_heads': 24,
                    'init_values': 1e-5,
                    'embed_dim': 1536,
                    'mlp_ratio': 2.66667 * 2,
                    'num_classes': 0,
                    'no_embed_class': True,
                    'mlp_layer': timm.layers.SwiGLUPacked,
                    'act_layer': nn.SiLU,
                    'reg_tokens': 8,
                    'dynamic_img_size': True
                }
                self.model = timm.create_model(pretrained=False, **timm_kwargs)
            else:
                print("Using random initialization with vit_large_patch16_224 for testing")
                self.model = timm.create_model(
                    "vit_large_patch16_224",
                    pretrained=False,
                    num_classes=0
                )

    def _get_transform(self) -> transforms.Compose:
        """UNI preprocessing transforms."""
        # Standard ImageNet normalization
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )


class Virchow2Encoder(Encoder):
    """
    Virchow2 encoder from Paige AI.
    Uses SwiGLU MLPs and outputs 2560-dim embeddings (class + patch tokens).
    """

    def __init__(self, device: str = "cuda", use_mixed_precision: bool = True):
        super().__init__(device, use_mixed_precision)
        self.embed_dim = 2560  # 1280 class token + 1280 mean patch tokens
        self.transform = self._get_transform()
        self._load_model()
        # Move model to device and set to eval mode
        if self.model is not None:
            self.model = self.model.to(self.device)
            self.model.eval()

    def _load_model(self):
        """Load Virchow2 model using official configuration."""
        try:
            # Official Virchow2 configuration from HuggingFace
            # https://huggingface.co/paige-ai/Virchow2
            self.model = timm.create_model(
                "hf-hub:paige-ai/Virchow2",
                pretrained=True,
                mlp_layer=timm.layers.SwiGLUPacked,
                act_layer=nn.SiLU
            )
            print(f"Info: Loaded Virchow2 with official timm configuration")

        except Exception as e:
            print(f"Warning: Could not load Virchow2 from HuggingFace: {e}")
            print("Using random initialization for testing purposes")
            self.model = timm.create_model(
                "vit_huge_patch14_224",
                pretrained=False,
                num_classes=0,
                mlp_layer=timm.layers.SwiGLUPacked,
                act_layer=nn.SiLU
            )

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> np.ndarray:
        """
        Extract Virchow2 embeddings: concatenate class token + mean patch tokens.

        Virchow2 outputs: [batch, 261, 1280]
        - Token 0: class token
        - Tokens 1-4: register tokens (ignored)
        - Tokens 5-260: patch tokens (256 patches)

        Final embedding: [batch, 2560] = class token (1280) + mean patch tokens (1280)
        """
        if self.model is None:
            self._load_model()
            self.model = self.model.to(self.device)
            self.model.eval()

        # Preprocess
        images = self.preprocess(images).to(self.device)

        # Forward pass with mixed precision
        if self.use_mixed_precision:
            with torch.cuda.amp.autocast():
                output = self.model(images)  # [B, 261, 1280]
        else:
            output = self.model(images)

        # Extract class token and patch tokens
        class_token = output[:, 0]  # [B, 1280]
        patch_tokens = output[:, 5:]  # [B, 256, 1280] - skip register tokens

        # Concatenate class token and mean of patch tokens
        embeddings = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # [B, 2560]

        return embeddings.cpu().numpy().astype(np.float32)

    def _get_transform(self) -> transforms.Compose:
        """Virchow2 preprocessing transforms."""
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )


class HOptimus0Encoder(Encoder):
    """
    H-Optimus-0 encoder from Bioptimus.
    1536-dim foundation model for histopathology at 0.5 μm/px.
    """

    def __init__(self, device: str = "cuda", use_mixed_precision: bool = True):
        super().__init__(device, use_mixed_precision)
        self.embed_dim = 1536
        self.transform = self._get_transform()
        self._load_model()
        # Move model to device and set to eval mode
        if self.model is not None:
            self.model = self.model.to(self.device)
            self.model.eval()

    def _load_model(self):
        """Load H-Optimus-0 model using official configuration."""
        try:
            # Official H-Optimus-0 configuration from HuggingFace
            # https://huggingface.co/bioptimus/H-optimus-0
            self.model = timm.create_model(
                "hf-hub:bioptimus/H-optimus-0",
                pretrained=True,
                init_values=1e-5,
                dynamic_img_size=False
            )
            print(f"Info: Loaded H-Optimus-0 with official timm configuration")

        except Exception as e:
            print(f"Warning: Could not load H-Optimus-0 from HuggingFace: {e}")
            print("Using random initialization for testing purposes")
            self.model = timm.create_model(
                "vit_large_patch14_224",
                pretrained=False,
                num_classes=0,
                init_values=1e-5
            )

    def _get_transform(self) -> transforms.Compose:
        """H-Optimus-0 preprocessing transforms (expects 0.5 μm/px resolution)."""
        return transforms.Normalize(
            mean=[0.707223, 0.578729, 0.703617],
            std=[0.211883, 0.230117, 0.177517]
        )


def get_encoder(encoder_name: str, model_id: Optional[str] = None,
                device: str = "cuda", use_mixed_precision: bool = True) -> Encoder:
    """
    Factory function to create encoder by name.

    Args:
        encoder_name: Name of encoder ('uni', 'virchow2', 'hoptimus0')
        model_id: HuggingFace model ID (optional, uses default if not provided)
        device: Device to run on
        use_mixed_precision: Whether to use mixed precision

    Returns:
        Encoder instance

    Raises:
        ValueError: If encoder name is not recognized
    """
    encoder_name = encoder_name.lower()

    if encoder_name == 'uni':
        return UNIEncoder(model_id=model_id or "MahmoodLab/UNI",
                         device=device, use_mixed_precision=use_mixed_precision)
    elif encoder_name == 'virchow2':
        return Virchow2Encoder(device=device, use_mixed_precision=use_mixed_precision)
    elif encoder_name == 'hoptimus0':
        return HOptimus0Encoder(device=device, use_mixed_precision=use_mixed_precision)
    else:
        raise ValueError(
            f"Unknown encoder: {encoder_name}. "
            f"Available: ['uni', 'virchow2', 'hoptimus0']"
        )
