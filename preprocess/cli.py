"""Serial multi-species PlantGeneAnn preprocessing CLI."""
from __future__ import annotations
import argparse
import csv
import json
import logging
from pathlib import Path
from .config import load_config
from .discovery import discover_species
from .process_runtime import configure_multiprocess_resource_tracker


def parse_args():
    parser = argparse.ArgumentParser(description="Build PlantGeneAnn training Datasets from species FASTA/GFF folders.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stop_on_error", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # Configure the datasets/multiprocess runtime before importing the pipeline,
    # whose writer imports Hugging Face datasets.
    configure_multiprocess_resource_tracker()
    from .pipeline import process_species
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    logger = logging.getLogger("PlantGeneAnn.preprocess")
    config = load_config(args.config)
    resolved_workers = config.workers.resolved()
    if resolved_workers != config.workers:
        logger.warning("Worker counts exceeded available CPUs and were reduced to %s", resolved_workers)
        object.__setattr__(config, "workers", resolved_workers)
    model_path = str(Path(args.model_path).expanduser().resolve())
    if not Path(model_path).is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    species_inputs = discover_species(args.input_dir)
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifests, failures = [], []
    for index, item in enumerate(species_inputs, 1):
        logger.info("Processing species %d/%d: %s", index, len(species_inputs), item.name)
        try:
            manifests.append(process_species(item, output_root, model_path, config, args.overwrite))
        except Exception as error:
            logger.exception("Species failed: %s", item.name)
            failures.append((item.name, repr(error)))
            if args.stop_on_error:
                break
    with (output_root / "preprocess_batch_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"format_version": 1, "species_total": len(species_inputs), "succeeded": manifests, "failed": [{"species": x, "error": y} for x, y in failures]}, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (output_root / "preprocess_failures.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("species", "error")); writer.writerows(failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
