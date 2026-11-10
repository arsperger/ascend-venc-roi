"""
Backend-agnostic encoder interface and configuration.

Concrete backends live in their own modules — see
:mod:`ascend_venc_roi.ascend_encoder`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from types import TracebackType

import numpy as np
import numpy.typing as npt

from ascend_venc_roi.types import ROIRegion


class RateControl(Enum):
    """Rate-control strategy requested from the encoder backend."""

    #: Constant bitrate — stable bandwidth, quality varies with scene load.
    CBR = "cbr"
    #: Variable bitrate — ``bitrate_kbps`` is treated as the ceiling.
    VBR = "vbr"


@dataclass
class EncoderConfig:
    """
    Encoder configuration.

    Args:
        output_path: Destination file for the elementary bitstream.
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Frame rate, used for rate-control pacing.
        bitrate_kbps: Target bitrate (CBR) or ceiling (VBR), in kbit/s.
        rate_control: :class:`RateControl` strategy.
        gop: I-frame interval in frames.
        qp_min: Lower bound of the quantisation parameter the backend may use.
        qp_max: Upper bound of the quantisation parameter the backend may use.

    Note:
        ``qp_min`` matters more than it looks. Backend rate controllers
        commonly default to a floor around 24, which silently clamps any ROI
        request for higher quality and makes ROI appear to do nothing. The
        default of 10 here leaves the ROI range usable.

    Example:
        >>> EncoderConfig(output_path="out.h265", width=1280, height=720).gop
        30
    """

    output_path: str
    width: int
    height: int
    fps: float = 30.0

    bitrate_kbps: int = 2000
    rate_control: RateControl = RateControl.CBR
    gop: int = 30

    qp_min: int = 10
    qp_max: int = 51


@dataclass
class EncoderStats:
    """
    Encoder performance counters.

    Attributes:
        frames_written: Frames successfully encoded and written out.
        frames_dropped: Frames rejected or lost at any pipeline stage.
        bytes_written: Total bitstream bytes written.
        encoding_time_ms: Cumulative wall-clock time spent in ``write_frame``.
        avg_fps: Frames written divided by ``encoding_time_ms``.
        roi_updates: Frames for which a non-empty region list was supplied.
    """

    frames_written: int = 0
    frames_dropped: int = 0
    bytes_written: int = 0
    encoding_time_ms: float = 0.0
    avg_fps: float = 0.0
    roi_updates: int = 0


class IEncoder(ABC):
    """
    Abstract encoder backend.

    Callers depend on this interface rather than on a concrete backend, so
    that swapping hardware does not change call sites. Implementations must
    support the context-manager protocol, since every known backend owns
    resources (device channels, file handles) that leak if not released.

    Example:
        >>> def encode(encoder: IEncoder, stream) -> None:
        ...     with encoder:
        ...         for frame, regions in stream:
        ...             encoder.write_frame(frame, regions)
    """

    @abstractmethod
    def open(self) -> bool:
        """
        Acquire resources and prepare to accept frames.

        Returns:
            True if the encoder is ready.
        """

    @abstractmethod
    def close(self) -> None:
        """Flush, finalise the output, and release all resources."""

    @abstractmethod
    def write_frame(
        self,
        frame: npt.NDArray[np.uint8],
        regions: list[ROIRegion] | None = None,
    ) -> bool:
        """
        Encode a single frame, optionally with regions of interest.

        Args:
            frame: BGR frame, ``(H, W, 3)``, dtype ``uint8``.
            regions: Regions to prioritise in this frame. Backends may
                impose their own count and alignment limits.

        Returns:
            True if the frame was encoded and written.
        """

    @property
    @abstractmethod
    def stats(self) -> EncoderStats:
        """Current encoder statistics."""

    @property
    @abstractmethod
    def is_open(self) -> bool:
        """Whether the encoder is open and accepting frames."""

    def __enter__(self) -> IEncoder:
        """Open the encoder, raising if it cannot be opened."""
        if not self.open():
            raise RuntimeError(f"Failed to open {type(self).__name__}")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the encoder."""
        self.close()
