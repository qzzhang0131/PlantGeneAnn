"""Shared helpers for PlantGeneAnn command-line pipelines."""

from __future__ import annotations

import errno
import glob
import logging
import math
import os
import signal
import shutil
import subprocess
import time
from typing import Mapping, Sequence, Tuple

logger = logging.getLogger("PlantGeneAnn.src.pipeline_utils")

PREDICTION_NUM_STRANDS = 2
PREDICTION_NUM_CLASSES = 5
FLOAT16_BYTES = 2
DEFAULT_DISK_SAFETY_FACTOR = 1.20
RUNTIME_CACHE_RETRY_DELAYS = (0.2, 0.5, 1.0)
SUBPROCESS_TERMINATION_GRACE_SECONDS = 5.0


def _remove_runtime_cache_path(path: str) -> bool:
    """Remove one cache path, retrying transient NFS busy/non-empty errors."""

    for attempt in range(len(RUNTIME_CACHE_RETRY_DELAYS) + 1):
        try:
            if os.path.islink(path) or os.path.isfile(path):
                os.remove(path)
                return True
            if os.path.isdir(path):
                shutil.rmtree(path)
                return True
            return False
        except FileNotFoundError:
            # Another cleanup path may already have removed this entry.
            return False
        except OSError as error:
            can_retry = error.errno in {errno.EBUSY, errno.ENOTEMPTY}
            if can_retry and attempt < len(RUNTIME_CACHE_RETRY_DELAYS):
                time.sleep(RUNTIME_CACHE_RETRY_DELAYS[attempt])
                continue
            raise

    return False


def cleanup_prediction_runtime_cache(
    cache_path: str,
    *,
    keep_datasets: bool = False,
) -> int:
    """Remove non-persistent files created by prediction preprocessing.

    ``huggingface/`` is always a disposable runtime cache and is removed after
    both successful and failed runs. When ``keep_datasets`` is false, this also
    removes HuggingFace Datasets/Arrow caches, tokenized ``chunk_N`` datasets,
    tokenization shards, and pre-tokenization ``chunk_N.tsv`` files.

    Cleanup is best-effort so an error deleting one cache path never masks the
    original pipeline exception. The completed chromosome HDF5 is deliberately
    outside this function's scope.

    Returns:
        Number of cache paths successfully removed.
    """

    cache_path = os.path.abspath(cache_path)
    exact_names = ["huggingface"]
    patterns = []
    if not keep_datasets:
        exact_names.extend(["datasets", "shards"])
        patterns.append("chunk_*")

    candidate_paths = {
        os.path.join(cache_path, name) for name in exact_names
    }
    for pattern in patterns:
        candidate_paths.update(glob.glob(os.path.join(cache_path, pattern)))

    removed_count = 0
    for path in sorted(candidate_paths):
        try:
            if _remove_runtime_cache_path(path):
                removed_count += 1
        except OSError as error:
            logger.warning("Could not remove runtime cache %s: %s", path, error)

    if removed_count:
        logger.info(
            "Removed %d non-persistent prediction cache path(s) from %s",
            removed_count,
            cache_path,
        )
    return removed_count


def _posix_process_group_exists(process_group_id: int) -> bool:
    """Return whether a POSIX process group still has at least one member."""

    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # The group exists even if an unexpected permission boundary prevents
        # signalling it. The caller will still attempt normal process cleanup.
        return True
    return True


def _terminate_subprocess_group(
    process: subprocess.Popen,
    *,
    interrupt: bool = False,
    grace_seconds: float = SUBPROCESS_TERMINATION_GRACE_SECONDS,
) -> None:
    """Terminate a launcher and all descendant workers without masking errors."""

    if os.name == "posix":
        # The child is started with start_new_session=True, so its PID is also
        # the process-group ID. This remains usable after the launcher exits as
        # long as an orphaned inference/DataLoader descendant is still alive.
        process_group_id = process.pid
        first_signal = signal.SIGINT if interrupt else signal.SIGTERM
        try:
            os.killpg(process_group_id, first_signal)
        except ProcessLookupError:
            return
        except PermissionError as error:
            logger.warning(
                "Could not signal inference process group %d: %s",
                process_group_id,
                error,
            )
            return

        deadline = time.monotonic() + max(0.0, float(grace_seconds))
        while time.monotonic() < deadline:
            # poll() also reaps the launcher if it has already exited, so its
            # zombie entry cannot keep the process group artificially alive.
            process.poll()
            if not _posix_process_group_exists(process_group_id):
                return
            time.sleep(0.1)

        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            return
        except PermissionError as error:
            logger.warning(
                "Could not force-kill inference process group %d: %s",
                process_group_id,
                error,
            )
        finally:
            if process.poll() is None:
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass
        return

    # PlantGeneAnn inference targets Linux HPC systems, but retain a safe
    # single-process fallback for development environments without POSIX groups.
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=max(0.0, float(grace_seconds)))
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_accelerate_subprocess(command: Sequence[str]) -> None:
    """Run Accelerate in an isolated process group and reap it on every failure.

    A launcher can exit after its main inference process is OOM-killed while
    DataLoader descendants remain alive and keep Arrow files open. Isolating the
    whole launch in a process group lets the parent terminate those descendants
    before runtime-cache cleanup begins.
    """

    if not command:
        raise ValueError("Accelerate command must not be empty.")

    popen_kwargs = {}
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    elif os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    process = subprocess.Popen(list(command), **popen_kwargs)
    previous_sigterm_handler = None

    # Convert scheduler SIGTERM into SystemExit while waiting. This allows the
    # process-group teardown here and the caller's HDF5/cache finally blocks to
    # run before the parent exits with the conventional 128 + signal code.
    if os.name == "posix":
        previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

        def _handle_sigterm(signum, _frame):
            raise SystemExit(128 + int(signum))

        signal.signal(signal.SIGTERM, _handle_sigterm)

    try:
        try:
            returncode = process.wait()
        except BaseException as error:
            _terminate_subprocess_group(
                process,
                interrupt=isinstance(error, KeyboardInterrupt),
            )
            raise

        if returncode != 0:
            # The launcher may already be gone, but its process group can still
            # contain orphaned DataLoader workers holding Arrow files open.
            _terminate_subprocess_group(process)
            raise subprocess.CalledProcessError(returncode, list(command))
        if os.name == "posix" and _posix_process_group_exists(process.pid):
            # A successful launcher should leave no descendants. Reap any
            # unexpected worker that outlived it before the caller removes
            # Arrow datasets.
            _terminate_subprocess_group(process)
    finally:
        if os.name == "posix" and previous_sigterm_handler is not None:
            signal.signal(signal.SIGTERM, previous_sigterm_handler)


def _format_bytes(num_bytes: int) -> str:
    """Format a byte count using binary units for preflight log messages."""

    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def _nearest_existing_directory(path: str) -> str:
    """Return the nearest existing directory containing *path*."""

    candidate = os.path.abspath(path)
    if not os.path.isdir(candidate):
        candidate = os.path.dirname(candidate)

    while candidate and not os.path.exists(candidate):
        parent = os.path.dirname(candidate)
        if parent == candidate:
            break
        candidate = parent

    if not candidate or not os.path.isdir(candidate):
        raise FileNotFoundError(
            f"Cannot locate an existing parent directory for disk check: {path}"
        )
    return candidate


def estimate_prediction_disk_bytes(
    chrom_sequence_info: Mapping[str, Tuple[int, int]],
) -> int:
    """Estimate uncompressed direct chromosome-level probability bytes."""

    total_genomic_bases = sum(
        int(chrom_length) for chrom_length, _ in chrom_sequence_info.values()
    )
    bytes_per_predicted_base = (
        PREDICTION_NUM_STRANDS * PREDICTION_NUM_CLASSES * FLOAT16_BYTES
    )

    return total_genomic_bases * bytes_per_predicted_base


def ensure_prediction_disk_space(
    *,
    chromosome_h5_path: str,
    chrom_sequence_info: Mapping[str, Tuple[int, int]],
    safety_factor: float = DEFAULT_DISK_SAFETY_FACTOR,
) -> None:
    """Fail before inference when the single direct-write HDF5 will not fit."""

    if safety_factor < 1.0:
        raise ValueError(f"safety_factor must be at least 1.0, got {safety_factor}.")

    chromosome_h5_bytes = estimate_prediction_disk_bytes(chrom_sequence_info)
    output_probe = _nearest_existing_directory(chromosome_h5_path)
    required_bytes = int(math.ceil(chromosome_h5_bytes * safety_factor))
    free_bytes = int(shutil.disk_usage(output_probe).free)
    reclaimable_bytes = sum(
        int(os.path.getsize(path))
        for path in (chromosome_h5_path, f"{chromosome_h5_path}.tmp")
        if os.path.isfile(path)
    )
    available_after_replace = free_bytes + reclaimable_bytes

    logger.info(
        "Disk preflight for %s: required with margin=%s, available=%s",
        output_probe,
        _format_bytes(required_bytes),
        _format_bytes(available_after_replace),
    )

    if available_after_replace < required_bytes:
        raise OSError(
            "Insufficient disk space for direct chromosome-level predictions: "
            f"filesystem at {output_probe} requires {_format_bytes(required_bytes)} "
            f"with a {safety_factor - 1.0:.0%} safety margin, but only "
            f"{_format_bytes(available_after_replace)} will be available after "
            "replacing the old output."
        )


def detect_num_processes() -> int:
    """Return the number of local GPU worker processes to use for inference."""

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        visible_devices = [
            device.strip()
            for device in cuda_visible_devices.split(",")
            if device.strip() and device.strip() != "-1"
        ]
        return max(1, len(visible_devices))

    try:
        import torch

        gpu_count = torch.cuda.device_count()
        if gpu_count > 0:
            return gpu_count
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            gpu_count = sum(
                1 for line in result.stdout.splitlines()
                if line.strip().startswith("GPU ")
            )
            if gpu_count > 0:
                return gpu_count
    except Exception:
        pass

    return 1
