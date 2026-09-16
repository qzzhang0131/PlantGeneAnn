"""Shared logging, cache, subprocess, GPU, and disk runtime support."""

from __future__ import annotations

import errno
import glob
import gzip
import hashlib
import logging
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Sequence, Tuple

CPU_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}
for _thread_env_name, _thread_env_value in CPU_THREAD_ENVIRONMENT.items():
    os.environ[_thread_env_name] = _thread_env_value

from .constants import PREDICTION_NUM_CLASSES, PREDICTION_NUM_STRANDS

logger = logging.getLogger("PlantGeneAnn.src.runtime")

FLOAT16_BYTES = 2
DEFAULT_DISK_SAFETY_FACTOR = 1.20
RUNTIME_CACHE_RETRY_DELAYS = (0.2, 0.5, 1.0)
SUBPROCESS_TERMINATION_GRACE_SECONDS = 5.0
MAX_DATALOADER_WORKERS_PER_RANK = 4
FASTA_INDEX_CACHE_DIRNAME = "faidx"
MATERIALIZED_FASTA_CACHE_DIRNAME = "fasta"
GZIP_MAGIC = b"\x1f\x8b"
STREAM_COPY_BUFFER_BYTES = 8 * 1024 * 1024
DEFAULT_PREDICTION_H5_FILENAME = "chromosome_predictions.h5"
DEFAULT_PREDICTION_OUTPUT_ENTRIES = (
    DEFAULT_PREDICTION_H5_FILENAME,
    f"{DEFAULT_PREDICTION_H5_FILENAME}.tmp",
)


class FastaRuntimePolicy(str, Enum):
    """Control whether FASTA runtime files survive successful prediction."""

    CLEAN_AFTER_PREDICTION = "clean_after_prediction"
    RETAIN_FOR_DECODING = "retain_for_decoding"


@dataclass(frozen=True)
class CpuResourcePlan:
    """Resolved CPU budget shared by preprocessing and distributed inference."""

    total_threads: int
    accelerate_processes: int
    dataloader_workers_per_rank: int


def detect_available_cpu_threads() -> int:
    """Return the CPU count available to this process under affinity limits."""

    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, int(os.cpu_count() or 1))


def resolve_num_cpu_threads(num_cpu_threads: Optional[int]) -> int:
    """Resolve a requested CPU budget without exceeding process affinity."""

    available = detect_available_cpu_threads()
    if num_cpu_threads is None:
        return available

    requested = int(num_cpu_threads)
    if requested <= 0:
        raise ValueError(
            f"num_cpu_threads must be positive, got {requested}."
        )
    if requested > available:
        logger.warning(
            "Requested %d CPU threads, but only %d are available; using %d.",
            requested,
            available,
            available,
        )
        return available
    return requested


def resolve_dataloader_workers_per_rank(
    num_cpu_threads: int,
    num_processes: int,
) -> int:
    """Allocate DataLoader workers per Accelerate rank from one CPU budget."""

    total_threads = int(num_cpu_threads)
    processes = int(num_processes)
    if total_threads <= 0:
        raise ValueError("num_cpu_threads must be positive.")
    if processes <= 0:
        raise ValueError("num_processes must be positive.")

    workers = max(0, (total_threads - processes) // processes)
    return min(MAX_DATALOADER_WORKERS_PER_RANK, workers)


def build_cpu_resource_plan(
    *,
    num_cpu_threads: Optional[int],
    num_processes: int,
) -> CpuResourcePlan:
    """Build the shared preprocessing/inference CPU allocation."""

    total_threads = resolve_num_cpu_threads(num_cpu_threads)
    processes = int(num_processes)
    if processes <= 0:
        raise ValueError("num_processes must be positive.")
    return CpuResourcePlan(
        total_threads=total_threads,
        accelerate_processes=processes,
        dataloader_workers_per_rank=resolve_dataloader_workers_per_rank(
            total_threads,
            processes,
        ),
    )


def configure_single_threaded_libraries() -> None:
    """Prevent nested native thread pools inside multiprocessing workers."""

    for name, value in CPU_THREAD_ENVIRONMENT.items():
        os.environ[name] = value

    try:
        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch allows setting inter-op threads only before parallel work
            # starts. Environment limits still protect already-started runtimes.
            pass
    except ImportError:
        pass


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

def _is_gzip_file(path: str) -> bool:
    """Return whether a regular file starts with the gzip magic bytes."""

    with open(path, "rb") as handle:
        return handle.read(len(GZIP_MAGIC)) == GZIP_MAGIC


def _materialized_fasta_path(compressed_path: str, cache_path: str) -> str:
    """Return a cache path bound to one immutable view of a gzip input.

    Path, size, and nanosecond modification time prevent a reused cache from
    silently serving an older decompression after the source file changes.
    """

    source_path = os.path.realpath(os.path.abspath(compressed_path))
    source_stat = os.stat(source_path)
    identity = "\0".join(
        (
            os.path.normcase(source_path),
            str(source_stat.st_size),
            str(source_stat.st_mtime_ns),
        )
    )
    digest = hashlib.sha256(os.fsencode(identity)).hexdigest()
    return os.path.join(
        os.path.abspath(cache_path),
        MATERIALIZED_FASTA_CACHE_DIRNAME,
        f"{digest}.fa",
    )


def prepare_genome_fasta(fasta_path: str, cache_path: str) -> str:
    """Return a random-access-compatible FASTA path for plain or gzip input.

    Plain FASTA files are returned unchanged. A gzip stream is decompressed in
    bounded-memory blocks into a deterministic cache-local file. The temporary
    output is atomically promoted only after the gzip footer has been read and
    validated, so truncated/corrupt inputs cannot leave a reusable partial FASTA.
    Compression is detected from file content; conventional ``.gz`` names whose
    contents are not gzip are rejected with a focused error.
    """

    source_path = os.path.abspath(fasta_path)
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Genome FASTA not found: {source_path}")

    is_gzip = _is_gzip_file(source_path)
    if not is_gzip:
        if source_path.lower().endswith(".gz"):
            raise ValueError(
                f"Genome file has a .gz extension but is not valid gzip data: {source_path}"
            )
        return source_path

    output_path = _materialized_fasta_path(source_path, cache_path)
    if os.path.isfile(output_path):
        logger.debug("Reusing materialized gzip FASTA: %s", output_path)
        return output_path
    if os.path.isdir(output_path):
        raise IsADirectoryError(
            f"Materialized FASTA path unexpectedly refers to a directory: {output_path}"
        )

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    temporary_path = None
    logger.info("Decompressing genome FASTA into runtime cache: %s", source_path)
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".fasta_",
            suffix=".tmp",
            dir=output_dir,
            delete=False,
        ) as output_handle:
            temporary_path = output_handle.name
            with gzip.open(source_path, "rb") as input_handle:
                shutil.copyfileobj(
                    input_handle,
                    output_handle,
                    length=STREAM_COPY_BUFFER_BYTES,
                )
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    except (OSError, EOFError) as error:
        raise IOError(
            f"Failed to decompress genome FASTA {source_path}: {error}"
        ) from error
    finally:
        if temporary_path is not None:
            try:
                os.remove(temporary_path)
            except FileNotFoundError:
                pass

    logger.info("Genome FASTA decompressed: %s", output_path)
    return output_path


def cleanup_materialized_genome_fasta(
    original_fasta_path: str,
    resolved_fasta_path: str,
    cache_path: str,
) -> bool:
    """Remove an exact cache-local decompression, never a user-owned FASTA."""

    original_path = os.path.abspath(original_fasta_path)
    resolved_path = os.path.abspath(resolved_fasta_path)
    cache_dir = os.path.join(
        os.path.abspath(cache_path), MATERIALIZED_FASTA_CACHE_DIRNAME
    )
    if resolved_path == original_path or os.path.dirname(resolved_path) != cache_dir:
        return False

    try:
        removed = _remove_runtime_cache_path(resolved_path)
    except OSError as error:
        logger.warning(
            "Could not remove materialized FASTA cache %s: %s",
            resolved_path,
            error,
        )
        return False
    try:
        os.rmdir(cache_dir)
    except (FileNotFoundError, OSError):
        pass
    return removed


def get_cached_fasta_index_path(fasta_path: str, cache_path: str) -> str:
    """Return the deterministic cache-local ``pyfaidx`` index path.

    The normalized absolute FASTA path is hashed so different input genomes can
    never reuse one another's index when a cache directory is reused. File
    contents are deliberately not hashed: ``pyfaidx`` already detects and
    rebuilds an index when its FASTA is newer.
    """

    normalized_fasta_path = os.path.normcase(
        os.path.realpath(os.path.abspath(fasta_path))
    )
    path_digest = hashlib.sha256(os.fsencode(normalized_fasta_path)).hexdigest()
    return os.path.join(
        os.path.abspath(cache_path),
        FASTA_INDEX_CACHE_DIRNAME,
        f"{path_digest}.fai",
    )


def prepare_cached_fasta_index_path(fasta_path: str, cache_path: str) -> str:
    """Create the reserved index-cache directory and return its ``.fai`` path."""

    index_path = get_cached_fasta_index_path(fasta_path, cache_path)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    return index_path


def cleanup_cached_fasta_index(fasta_path: str, cache_path: str) -> bool:
    """Best-effort removal of the cache index belonging to one input FASTA.

    Only the deterministic PlantGeneAnn cache file is removed. A user-owned
    ``<input>.fai`` beside the genome is never considered by this function.
    """

    index_path = get_cached_fasta_index_path(fasta_path, cache_path)
    try:
        removed = _remove_runtime_cache_path(index_path)
    except OSError as error:
        logger.warning("Could not remove FASTA index cache %s: %s", index_path, error)
        return False

    # Keep a shared faidx namespace when it still contains another cached
    # genome; otherwise remove the now-empty directory without broad deletion.
    try:
        os.rmdir(os.path.dirname(index_path))
    except (FileNotFoundError, OSError):
        pass
    return removed


def _cleanup_runtime_cache_entries(
    cache_path: str,
    *,
    exact_names: Sequence[str],
    patterns: Sequence[str] = (),
) -> int:
    """Best-effort removal for an explicit set of cache-local entries."""

    cache_path = os.path.abspath(cache_path)
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
    return removed_count


def cleanup_prediction_work_cache(cache_path: str) -> int:
    """Remove caches used only by extraction, tokenization, and inference."""

    return _cleanup_runtime_cache_entries(
        cache_path,
        exact_names=("huggingface", "datasets", "shards"),
        patterns=("chunk_*",),
    )


def cleanup_fasta_runtime_cache(cache_path: str) -> int:
    """Remove cache-local FASTA materializations and their indexes."""

    return _cleanup_runtime_cache_entries(
        cache_path,
        exact_names=(
            FASTA_INDEX_CACHE_DIRNAME,
            MATERIALIZED_FASTA_CACHE_DIRNAME,
        ),
    )


def cleanup_prediction_runtime_cache(cache_path: str) -> int:
    """Remove every non-persistent prediction runtime cache category.

    Cleanup is best-effort so an error deleting one cache path never masks the
    original pipeline exception. The completed chromosome HDF5 is deliberately
    outside this function's scope.

    Returns:
        Number of cache paths successfully removed.
    """

    return (
        cleanup_prediction_work_cache(cache_path)
        + cleanup_fasta_runtime_cache(cache_path)
    )


def get_default_prediction_h5_path(cache_path: str) -> str:
    """Return the canonical chromosome-prediction HDF5 path for one cache."""

    return os.path.join(
        os.path.abspath(cache_path),
        DEFAULT_PREDICTION_H5_FILENAME,
    )


def cleanup_prediction_artifacts(
    cache_path: str,
    *,
    remove_default_outputs: bool = False,
    best_effort: bool = True,
) -> int:
    """Remove shared prediction artifacts under an explicit cache directory.

    Runtime preprocessing caches are always delegated to
    :func:`cleanup_prediction_runtime_cache`. Completed/default prediction HDF5
    files are removed only when ``remove_default_outputs`` is true, so the
    two-step prediction command keeps its requested output by default.

    ``best_effort=False`` lets Python API callers request strict removal of
    default HDF5 outputs. Runtime-cache removal remains best-effort and never
    masks an earlier pipeline exception.
    """

    removed_count = cleanup_prediction_runtime_cache(cache_path)
    if not remove_default_outputs:
        return removed_count

    cache_path = os.path.abspath(cache_path)
    for entry_name in DEFAULT_PREDICTION_OUTPUT_ENTRIES:
        path = os.path.join(cache_path, entry_name)
        try:
            if _remove_runtime_cache_path(path):
                removed_count += 1
        except OSError as error:
            if not best_effort:
                raise
            logger.warning("Could not remove prediction artifact %s: %s", path, error)

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
    """Return the number of GPUs visible and usable by the current process.

    PyTorch is authoritative because its device count respects
    ``CUDA_VISIBLE_DEVICES`` and reflects the devices that model inference can
    actually use. Environment and ``nvidia-smi`` fallbacks are retained for
    environments where importing or querying PyTorch fails. A CPU-only run uses
    one process.
    """

    try:
        import torch

        gpu_count = int(torch.cuda.device_count())
        if gpu_count > 0:
            return gpu_count
        return 1
    except Exception:
        pass

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is not None:
        visible_devices = [
            device.strip()
            for device in cuda_visible_devices.split(",")
            if device.strip() and device.strip() != "-1"
        ]
        return max(1, len(visible_devices))

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


def resolve_num_processes(num_processes: Optional[int]) -> int:
    """Resolve an optional Accelerate process count to a positive integer."""

    resolved = detect_num_processes() if num_processes is None else int(num_processes)
    if resolved <= 0:
        raise ValueError(f"num_processes must be positive, got {resolved}.")
    return resolved


LOGGER_NAME = "PlantGeneAnn"
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"


def setup_logger(*, verbose: bool = False, enabled: bool = True) -> logging.Logger:
    """Configure and return the shared PlantGeneAnn console logger.

    Args:
        verbose: Emit DEBUG messages when true; otherwise emit INFO and above.
        enabled: Install the console handler when true. Accelerate non-main
            ranks pass false so only rank 0 writes pipeline logs.

    Reconfiguration is intentional and idempotent: entry points call this once
    after parsing arguments, while tests and embedded callers may call it more
    than once. Existing handlers are removed and closed before the new handler
    is installed, preventing duplicated messages.
    """

    logger = logging.getLogger(LOGGER_NAME)
    level = logging.DEBUG if verbose else logging.INFO

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    logger.setLevel(level)
    logger.propagate = False

    if enabled:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_handler.setFormatter(
            logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        )
        logger.addHandler(console_handler)
    else:
        # A NullHandler also prevents logging.lastResort from printing WARNING
        # messages emitted by child loggers on non-main Accelerate ranks.
        logger.addHandler(logging.NullHandler())

    return logger
