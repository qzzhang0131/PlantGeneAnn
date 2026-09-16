#!/usr/bin/env python3
"""Download one or more PlantGeneAnn v2 checkpoints from Hugging Face."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence


MODEL_REPOS = {
    "angiospermae": "qzzhang/PlantGeneAnn-v2-Angiospermae",
    "bryophyta": "qzzhang/PlantGeneAnn-v2-Bryophyta",
    "chlorophyta": "qzzhang/PlantGeneAnn-v2-Chlorophyta",
}
MODEL_DIR_NAMES = {
    "angiospermae": "PlantGeneAnn-v2-Angiospermae",
    "bryophyta": "PlantGeneAnn-v2-Bryophyta",
    "chlorophyta": "PlantGeneAnn-v2-Chlorophyta",
}
MODEL_TYPES = ("angiospermae", "bryophyta", "chlorophyta")
MODEL_TYPE_CHOICES = MODEL_TYPES + ("all",)
DEFAULT_MODEL_TYPE = "angiospermae"
DEFAULT_REPO_ID = MODEL_REPOS[DEFAULT_MODEL_TYPE]
DEFAULT_REVISION = "main"
OFFICIAL_ENDPOINT = "https://huggingface.co"
MIRROR_ENDPOINT = "https://hf-mirror.com"
REQUIRED_FILES = (
    "config.json",
    "configuration_caduceus_ph.py",
    "modeling_caduceus_moe.py",
    "modeling_segment_caduceus_v2.py",
    "tokenization_caduceus.py",
    "tokenizer_config.json",
    "special_tokens_map.json",
)

logger = logging.getLogger("PlantGeneAnn")


def default_output_dir(model_type: str = DEFAULT_MODEL_TYPE) -> str:
    """Return the default local directory for a selected model type.

    ``all`` returns the model root; each official model is downloaded into a
    named child directory below that root.
    """

    models_root = Path(__file__).resolve().parent / "models"
    if model_type == "all":
        return str(models_root)
    if model_type not in MODEL_DIR_NAMES:
        valid = ", ".join(MODEL_TYPE_CHOICES)
        raise ValueError(
            f"unknown model_type {model_type!r}; expected one of: {valid}"
        )
    return str(models_root / MODEL_DIR_NAMES[model_type])


def validate_model(model_dir: str) -> None:
    """Check that the downloaded directory contains a usable model snapshot."""

    root = Path(model_dir)
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    has_weights = any(
        (root / name).is_file()
        for name in (
            "pytorch_model.bin",
            "model.safetensors",
            "pytorch_model.bin.index.json",
            "model.safetensors.index.json",
        )
    )
    if missing or not has_weights:
        details = []
        if missing:
            details.append("missing files: " + ", ".join(missing))
        if not has_weights:
            details.append("model weights are missing")
        raise RuntimeError("Incomplete model download: " + "; ".join(details))


def download_model(
    repo_id: str,
    output_dir: str,
    *,
    revision: str = DEFAULT_REVISION,
    endpoint: str = "auto",
    token: Optional[str] = None,
    force: bool = False,
    snapshot_downloader: Optional[Callable[..., str]] = None,
) -> str:
    """Download a complete model, falling back to HF Mirror in auto mode."""

    output_dir = str(Path(output_dir).expanduser().resolve())
    if not force and Path(output_dir).is_dir():
        try:
            validate_model(output_dir)
        except (OSError, ValueError, RuntimeError):
            pass
        else:
            logger.info("Model already exists: %s", output_dir)
            return output_dir

    if snapshot_downloader is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise RuntimeError(
                "huggingface_hub is required; install it with "
                "'pip install huggingface_hub'."
            ) from error
        snapshot_downloader = snapshot_download

    endpoints = {
        "auto": (OFFICIAL_ENDPOINT, MIRROR_ENDPOINT),
        "official": (OFFICIAL_ENDPOINT,),
        "mirror": (MIRROR_ENDPOINT,),
    }.get(endpoint)
    if endpoints is None:
        raise ValueError("endpoint must be 'auto', 'official', or 'mirror'.")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    errors = []
    for current_endpoint in endpoints:
        logger.info("Downloading from %s...", current_endpoint)
        try:
            snapshot_downloader(
                repo_id=repo_id,
                repo_type="model",
                revision=revision,
                local_dir=output_dir,
                endpoint=current_endpoint,
                token=token,
                force_download=force,
                max_workers=4,
            )
            validate_model(output_dir)
            logger.info("Model downloaded successfully: %s", output_dir)
            return output_dir
        except KeyboardInterrupt:
            raise
        except Exception as error:
            errors.append(f"{current_endpoint}: {error}")
            logger.warning("Download failed from %s: %s", current_endpoint, error)

    raise RuntimeError("All model download sources failed. " + " | ".join(errors))


def download_selected_models(
    model_type: str,
    output_dir: Optional[str] = None,
    *,
    repo_id: Optional[str] = None,
    revision: str = DEFAULT_REVISION,
    endpoint: str = "auto",
    token: Optional[str] = None,
    force: bool = False,
    snapshot_downloader: Optional[Callable[..., str]] = None,
) -> List[str]:
    """Download one selected model, or all three official models.

    For a single model, ``output_dir`` is the final model directory and an
    explicit ``repo_id`` is retained as a backwards-compatible override. For
    ``all``, ``output_dir`` is treated as a root and the three model-specific
    directories are created below it.
    """

    if model_type not in MODEL_TYPE_CHOICES:
        valid = ", ".join(MODEL_TYPE_CHOICES)
        raise ValueError(
            f"unknown model_type {model_type!r}; expected one of: {valid}"
        )

    if model_type == "all":
        if repo_id is not None:
            raise ValueError("--repo_id cannot be used with --model_type all")
        root = Path(output_dir or default_output_dir("all")).expanduser()
        downloaded = []
        download_kwargs: Dict[str, Any] = {
            "revision": revision,
            "endpoint": endpoint,
            "token": token,
            "force": force,
        }
        if snapshot_downloader is not None:
            download_kwargs["snapshot_downloader"] = snapshot_downloader
        for selected_type in MODEL_TYPES:
            model_dir = root / MODEL_DIR_NAMES[selected_type]
            try:
                downloaded.append(
                    download_model(
                        MODEL_REPOS[selected_type],
                        str(model_dir),
                        **download_kwargs,
                    )
                )
            except KeyboardInterrupt:
                raise
            except Exception as error:
                raise RuntimeError(
                    f"Failed to download {selected_type} model "
                    f"({MODEL_REPOS[selected_type]}): {error}"
                ) from error
        return downloaded

    model_dir = output_dir or default_output_dir(model_type)
    download_kwargs: Dict[str, Any] = {
        "revision": revision,
        "endpoint": endpoint,
        "token": token,
        "force": force,
    }
    if snapshot_downloader is not None:
        download_kwargs["snapshot_downloader"] = snapshot_downloader
    downloaded = download_model(
        repo_id or MODEL_REPOS[model_type],
        model_dir,
        **download_kwargs,
    )
    return [downloaded]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download PlantGeneAnn model files. Hugging Face is tried first and "
            "HF Mirror is used automatically if the official site is unavailable."
        )
    )
    parser.add_argument(
        "--model_type",
        choices=MODEL_TYPE_CHOICES,
        default=DEFAULT_MODEL_TYPE,
        help=(
            "Official model family to download. 'all' downloads all three "
            "families (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--repo_id",
        default=None,
        help=(
            "Optional Hugging Face repository override for a single model; "
            "cannot be combined with --model_type all."
        ),
    )
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--model_dir",
        default=None,
        help=(
            "Output directory. For one model this is the final model directory; "
            "for --model_type all it is the output root. Defaults to the "
            "model-specific directory under models/ (or models/ for all)."
        ),
    )
    parser.add_argument(
        "--endpoint",
        choices=("auto", "official", "mirror"),
        default="auto",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.model_dir is None:
        args.model_dir = default_output_dir(args.model_type)
    if args.repo_id is None and args.model_type != "all":
        args.repo_id = MODEL_REPOS[args.model_type]
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args() if argv is None else parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        model_paths = download_selected_models(
            args.model_type,
            args.model_dir,
            repo_id=args.repo_id,
            revision=args.revision,
            endpoint=args.endpoint,
            token=os.environ.get("HF_TOKEN"),
            force=args.force,
        )
    except KeyboardInterrupt:
        logger.warning("Download cancelled; run the command again to resume.")
        return 130
    except Exception as error:
        logger.error("Model download failed: %s", error)
        return 1

    for model_path in model_paths:
        print(model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
