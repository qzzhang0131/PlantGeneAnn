"""PlantGeneAnn package with lazily loaded public decoding interfaces."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["EmissionCalibration", "GenePrediction", "SegmentalDecoder"]

if TYPE_CHECKING:
    from .segmental_core import EmissionCalibration, GenePrediction
    from .segmental_decoder import SegmentalDecoder


def __getattr__(name: str) -> Any:
    """Load public decoder interfaces only when they are explicitly requested."""

    if name in {"EmissionCalibration", "GenePrediction"}:
        from .segmental_core import EmissionCalibration, GenePrediction

        value = {
            "EmissionCalibration": EmissionCalibration,
            "GenePrediction": GenePrediction,
        }[name]
    elif name == "SegmentalDecoder":
        from .segmental_decoder import SegmentalDecoder

        value = SegmentalDecoder
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy public names to interactive tools without importing them."""

    return sorted(set(globals()) | set(__all__))
