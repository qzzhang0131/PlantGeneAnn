"""Export GeneAnn tuning checkpoints as loss-free HF inference models."""

from __future__ import annotations

import argparse
import logging
import os
import shutil

from .config import FullTuningConfig, HalfTuningConfig, PeftTuningConfig, load_tuning_config
from .model import FullTuningModel, MambaLoRATuningModel, REMOTE_CODE_FILES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a GeneAnn PEFT bundle (merge LoRA) or full-tuning checkpoint "
            "as a plain loss-free SegmentCaduceusPh directory."
        )
    )
    parser.add_argument("--model_path", required=True, help="Base or source model path.")
    parser.add_argument(
        "--peft_path",
        default=None,
        help="LoRA bundle directory (best/peft). Required for peft or half.",
    )
    parser.add_argument(
        "--full_path",
        default=None,
        help="Full-tuning bundle (best/full) or model dir (best/model).",
    )
    parser.add_argument(
        "--tuning_config",
        default=None,
        help="Optional YAML to select export mode explicitly.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--safe_serialization", action="store_true")
    return parser.parse_args()


def _copy_plain_model(source_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    # Prefer HF weight files + remote code.
    for name in os.listdir(source_dir):
        src = os.path.join(source_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(output_dir, name))
    for filename in REMOTE_CODE_FILES:
        src = os.path.join(source_dir, filename)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(output_dir, filename))


def _has_hf_weights(model_dir: str) -> bool:
    """Return whether a directory contains plain HF bin or safetensors weights."""

    return any(
        os.path.isfile(os.path.join(model_dir, filename))
        for filename in ("pytorch_model.bin", "model.safetensors")
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.peft_path and args.full_path:
        raise ValueError("--peft_path and --full_path are mutually exclusive.")

    mode = None
    config = None
    if args.tuning_config:
        config = load_tuning_config(args.tuning_config)
        mode = config.method
        if mode in {"peft", "half"} and args.full_path:
            raise ValueError(
                f"method={mode!r} is incompatible with --full_path; use --peft_path."
            )
        if mode == "full" and args.peft_path:
            raise ValueError(
                "method='full' is incompatible with --peft_path; use --full_path."
            )
    elif args.peft_path:
        mode = "peft"
    elif args.full_path:
        mode = "full"
    else:
        raise ValueError(
            "Specify exactly one of --peft_path or --full_path, "
            "or pass --tuning_config with the matching checkpoint path."
        )

    if mode in {"peft", "half"}:
        if not args.peft_path:
            raise ValueError("--peft_path is required for peft/half export.")
        model = MambaLoRATuningModel.from_pretrained(
            args.model_path, peft_path=args.peft_path
        )
        summary = model.parameter_summary()
        logging.info(
            "Loaded PEFT model: trainable=%d, LoRA=%d, decoder=%d, total=%d",
            summary.trainable,
            summary.lora,
            summary.decoder,
            summary.total,
        )
        model.save_pretrained(
            args.output_dir, safe_serialization=args.safe_serialization
        )
        logging.info("Merged loss-free checkpoint saved to %s", args.output_dir)
        return

    if mode == "full":
        if not args.full_path:
            raise ValueError(
                "--full_path is required for full export. --model_path identifies "
                "the base architecture and must never be exported as tuned weights."
            )
        # Prefer an already-exported inference model directory.
        source = args.full_path
        model_dir = source
        if os.path.isdir(source):
            nested = os.path.join(source, "model")
            if _has_hf_weights(nested):
                model_dir = nested
            elif _has_hf_weights(source):
                model_dir = source
        if _has_hf_weights(model_dir) and os.path.isfile(
            os.path.join(model_dir, "config.json")
        ):
            _copy_plain_model(model_dir, args.output_dir)
            logging.info(
                "Copied full-tuning inference model from %s to %s",
                model_dir,
                args.output_dir,
            )
            return

        # Rebuild from base + full bundle weights.
        if config is None or not isinstance(config, FullTuningConfig):
            if args.full_path and os.path.isdir(args.full_path):
                config = FullTuningConfig.from_yaml(args.full_path)
            else:
                raise ValueError(
                    "Full export without a plain model/ directory requires "
                    "--tuning_config or a full/ bundle containing full_tuning_config.yml."
                )
        model = FullTuningModel.from_pretrained(
            args.model_path,
            full_config=config,
            resume_path=args.full_path,
        )
        summary = model.parameter_summary()
        logging.info(
            "Loaded full-tuning model: trainable=%d, backbone=%d, decoder=%d, total=%d",
            summary.trainable,
            summary.backbone,
            summary.decoder,
            summary.total,
        )
        model.save_pretrained(
            args.output_dir, safe_serialization=args.safe_serialization
        )
        logging.info("Full-tuning inference checkpoint saved to %s", args.output_dir)
        return

    raise ValueError(f"Unsupported export mode: {mode!r}")


if __name__ == "__main__":
    main()
