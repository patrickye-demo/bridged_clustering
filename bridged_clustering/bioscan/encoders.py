"""Encoder loading and feature extraction for BIOSCAN."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from transformers import AutoModel, AutoTokenizer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class EncoderSuite:
    barcode_tokenizer: Any
    barcode_model: Any
    image_model: Any
    image_transform: transforms.Compose


def _allow_remote_code() -> bool:
    return os.environ.get("BRIDGED_CLUSTERING_ALLOW_REMOTE_CODE", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def load_encoder_suite() -> EncoderSuite:
    """Load pretrained encoders used by the BIOSCAN experiments."""
    barcode_model_name = "bioscan-ml/BarcodeBERT"
    allow_remote_code = _allow_remote_code()
    try:
        barcode_tokenizer = AutoTokenizer.from_pretrained(
            barcode_model_name,
            trust_remote_code=allow_remote_code,
        )
        barcode_model = AutoModel.from_pretrained(
            barcode_model_name,
            trust_remote_code=allow_remote_code,
        ).to(DEVICE)
    except Exception as exc:  # pragma: no cover - depends on local HF state
        if not allow_remote_code and "trust_remote_code" in str(exc):
            raise RuntimeError(
                "BarcodeBERT requires remote code. Review the model repository first, then rerun with "
                "BRIDGED_CLUSTERING_ALLOW_REMOTE_CODE=1 to opt in explicitly.",
            ) from exc
        raise

    effnet_weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    image_model = efficientnet_b0(weights=effnet_weights).to(DEVICE)
    image_model.eval()

    image_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
    )
    return EncoderSuite(
        barcode_tokenizer=barcode_tokenizer,
        barcode_model=barcode_model,
        image_model=image_model,
        image_transform=image_transform,
    )


def load_pretrained_models():
    """Compatibility wrapper matching the original script interface."""
    suite = load_encoder_suite()
    return suite.barcode_tokenizer, suite.barcode_model, suite.image_model, suite.image_transform


def encode_images(
    image_ids: np.ndarray | list[str],
    image_paths: dict[str, str],
    model: Any,
    transform: transforms.Compose,
) -> np.ndarray:
    """Encode image ids into morphology embeddings."""
    features: list[np.ndarray] = []
    missing_ids: list[str] = []
    for process_id in image_ids:
        image_path = image_paths.get(process_id)
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = model(image_tensor)
            features.append(output.squeeze().cpu().numpy())
        else:
            missing_ids.append(str(process_id))
    if missing_ids:
        preview = ", ".join(missing_ids[:5])
        raise FileNotFoundError(
            "Missing BIOSCAN images for process ids: "
            f"{preview}{'...' if len(missing_ids) > 5 else ''}.",
        )
    return np.array(features)


def encode_genes(dna_barcodes: np.ndarray | list[str], tokenizer: Any, model: Any) -> np.ndarray:
    """Encode DNA barcode sequences into gene embeddings."""
    if isinstance(dna_barcodes, np.ndarray):
        dna_barcodes = [str(barcode) for barcode in dna_barcodes]

    embeddings: list[np.ndarray] = []
    for barcode in dna_barcodes:
        encodings = tokenizer(barcode, return_tensors="pt", padding=True, truncation=True)
        encodings = {key: value.unsqueeze(0).to(DEVICE) for key, value in encodings.items()}
        with torch.no_grad():
            embedding = model(**encodings).last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(embedding)

    array = np.array(embeddings)
    if len(array.shape) == 3:
        array = array.squeeze(1)
    return array


def encode_images_for_samples(
    df: pd.DataFrame,
    image_paths: dict[str, str],
    image_model: Any,
    image_transform: transforms.Compose,
) -> pd.DataFrame:
    features = encode_images(df["processid"].values, image_paths, image_model, image_transform)
    encoded = df.copy()
    encoded["morph_coordinates"] = features.tolist()
    return encoded


def encode_genes_for_samples(df: pd.DataFrame, barcode_tokenizer: Any, barcode_model: Any) -> pd.DataFrame:
    features = encode_genes(df["dna_barcode"].values, barcode_tokenizer, barcode_model)
    encoded = df.copy()
    encoded["gene_coordinates"] = features.tolist()
    return encoded
