"""
BGR→NV12 conversion on the Ascend VPC (Vision Preprocessing Core).

Performs the colour-space conversion on the NPU rather than in
``cv2.cvtColor``, and leaves the result in DVPP memory so that VENC can
consume it without a round trip through host memory.

The converter owns one VPC channel and a pair of reusable DVPP buffers
(BGR source, NV12 destination), so no allocation happens per frame.

Note:
    Frame memory for VENC must come from ``himpi.dvpp_malloc``; buffers from
    ``acl.rt.malloc`` are not accepted by the encoder.

Example:
    >>> with NV12Converter(1280, 720) as conv:  # doctest: +SKIP
    ...     buf = conv.convert(bgr_frame)
    ...     print(buf.addr, buf.size)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


@dataclass
class DVPPBuffer:
    """
    Handle to a device-side DVPP memory region.

    Attributes:
        addr: Device virtual address returned by ``dvpp_malloc``.
        size: Buffer size in bytes.
    """

    addr: int
    size: int


def _align16(val: int) -> int:
    """Round *val* up to the nearest multiple of 16."""
    return (val + 15) // 16 * 16


class NV12Converter:
    """
    BGR→NV12 converter backed by the Ascend VPC.

    Creates a VPC channel and allocates two DVPP buffers on :meth:`open`;
    releases both on :meth:`close`. :meth:`convert` uploads a BGR frame,
    runs the VPC colour-space kernel and returns a device-side NV12 buffer
    ready for VENC.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.
        vpc_chn: VPC channel id to use.

    Example:
        >>> with NV12Converter(1280, 720) as conv:  # doctest: +SKIP
        ...     buf = conv.convert(bgr_frame)
    """

    def __init__(self, width: int, height: int, *, vpc_chn: int = 0) -> None:
        self._width = width
        self._height = height
        self._vpc_chn = vpc_chn

        self._w_stride = _align16(width)
        self._bgr_size = width * height * 3
        self._nv12_size = self._w_stride * height * 3 // 2

        # Populated on open()
        self._bgr_addr: int = 0
        self._nv12_addr: int = 0
        self._opened = False
        self._frames_converted = 0

    # -- properties -------------------------------------------------------- #

    @property
    def nv12_size(self) -> int:
        """Size of a single NV12 frame in bytes."""
        return self._nv12_size

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return self._height

    @property
    def w_stride(self) -> int:
        """Width stride in bytes (width rounded up to 16)."""
        return self._w_stride

    @property
    def is_open(self) -> bool:
        """Whether the VPC channel and buffers are allocated."""
        return self._opened

    @property
    def frames_converted(self) -> int:
        """Number of frames converted since :meth:`open`."""
        return self._frames_converted

    # -- lifecycle --------------------------------------------------------- #

    def open(self) -> None:
        """
        Create the VPC channel and allocate DVPP buffers.

        Raises:
            RuntimeError: If the channel or either buffer cannot be created.
        """
        from acl import himpi

        # The empty dict as second argument is mandatory.
        ret: int = himpi.vpc_create_chn(self._vpc_chn, {})
        if ret != 0:
            raise RuntimeError(f"vpc_create_chn({self._vpc_chn}) failed: 0x{ret:08x}")

        self._bgr_addr = self._dvpp_alloc(self._bgr_size)
        self._nv12_addr = self._dvpp_alloc(self._nv12_size)

        self._opened = True
        logger.info(
            "NV12Converter opened: %dx%d stride=%d nv12=%d bytes (vpc_chn=%d)",
            self._width, self._height, self._w_stride, self._nv12_size, self._vpc_chn,
        )

    def close(self) -> None:
        """Release DVPP buffers and destroy the VPC channel."""
        if not self._opened:
            return
        from acl import himpi

        if self._nv12_addr:
            himpi.dvpp_free(self._nv12_addr)
            self._nv12_addr = 0
        if self._bgr_addr:
            himpi.dvpp_free(self._bgr_addr)
            self._bgr_addr = 0

        himpi.vpc_destroy_chn(self._vpc_chn)
        self._opened = False
        logger.info(
            "NV12Converter closed after %d frames (vpc_chn=%d)",
            self._frames_converted, self._vpc_chn,
        )

    def __enter__(self) -> NV12Converter:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- conversion -------------------------------------------------------- #

    def convert(self, frame: npt.NDArray[np.uint8]) -> DVPPBuffer:
        """
        Convert a BGR frame to NV12 on the NPU.

        Args:
            frame: ``(H, W, 3)`` BGR frame, dtype ``uint8``.

        Returns:
            :class:`DVPPBuffer` pointing at the NV12 data in DVPP memory.
            The buffer is reused on the next call — consume it before
            converting another frame.

        Raises:
            RuntimeError: If the converter is closed or a VPC call fails.
            ValueError: If *frame* has an unexpected shape or dtype.
        """
        if not self._opened:
            raise RuntimeError("NV12Converter is not open")

        h, w = frame.shape[:2]
        if (w, h) != (self._width, self._height):
            raise ValueError(
                f"Frame size {w}x{h} does not match converter {self._width}x{self._height}"
            )
        if frame.dtype != np.uint8:
            raise ValueError(f"Expected uint8 frame, got {frame.dtype}")

        import acl
        from acl import himpi

        # Host→device upload of the BGR frame.
        # NumPy 2.x: bytes_to_ptr, not the removed numpy_to_ptr.
        bgr_bytes = frame.tobytes()
        host_ptr = acl.util.bytes_to_ptr(bgr_bytes)
        ret: int = acl.rt.memcpy(
            self._bgr_addr, self._bgr_size, host_ptr, len(bgr_bytes), 1,
        )  # 1 = host to device
        if ret != 0:
            raise RuntimeError(f"BGR H2D memcpy failed: {ret}")

        source_pic: dict[str, Any] = {
            "picture_width": self._width,
            "picture_height": self._height,
            "picture_width_stride": self._width * 3,
            "picture_height_stride": self._height,
            "picture_format": 13,  # BGR_888
            "picture_address": self._bgr_addr,
            "picture_buffer_size": self._bgr_size,
        }

        dest_pic: dict[str, Any] = {
            "picture_width": self._width,
            "picture_height": self._height,
            "picture_width_stride": self._w_stride,
            "picture_height_stride": self._height,
            "picture_format": 1,  # NV12
            "picture_address": self._nv12_addr,
            "picture_buffer_size": self._nv12_size,
        }

        task_id, ret = himpi.vpc_convert_color_to_yuv420(
            self._vpc_chn, source_pic, dest_pic, -1,
        )
        if ret != 0:
            raise RuntimeError(f"vpc_convert_color_to_yuv420 failed: 0x{ret:08x}")

        proc_ret: int = himpi.vpc_get_process_result(self._vpc_chn, task_id, -1)
        if proc_ret != 0:
            raise RuntimeError(f"vpc_get_process_result failed: 0x{proc_ret:08x}")

        self._frames_converted += 1
        return DVPPBuffer(addr=self._nv12_addr, size=self._nv12_size)

    # -- internals --------------------------------------------------------- #

    @staticmethod
    def _dvpp_alloc(size: int) -> int:
        """Allocate DVPP memory and return the device address."""
        from acl import himpi

        result = himpi.dvpp_malloc(0, size)
        if isinstance(result, tuple):
            addr, ret = result
            if ret != 0:
                raise RuntimeError(f"dvpp_malloc({size}) failed: 0x{ret:08x}")
        else:
            addr = result
        return int(addr)
