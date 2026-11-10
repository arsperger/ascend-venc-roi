"""
Region-of-interest primitives shared by all encoder backends.

This module is deliberately free of any detector, codec or vendor coupling:
a caller supplies rectangles and quality offsets, and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Lower bound of the :attr:`ROIRegion.qoffset` range (maximum quality boost).
QOFFSET_MIN = -1.0

#: Upper bound of the :attr:`ROIRegion.qoffset` range (maximum quality penalty).
QOFFSET_MAX = 1.0


@dataclass
class ROIRegion:
    """
    A rectangular region with a relative quality offset.

    Coordinates are in frame pixel space with the origin at the top-left
    corner. Backends are free to align, clamp or drop regions that their
    hardware cannot express — see the backend documentation for the exact
    rules (the Ascend VENC backend aligns to 16 pixels and accepts at most
    8 regions per frame).

    Args:
        x: Left edge, in pixels from the frame's left edge.
        y: Top edge, in pixels from the frame's top edge.
        w: Region width in pixels.
        h: Region height in pixels.
        qoffset: Quality offset in ``[-1.0, +1.0]``. **Negative means better
            quality** (fewer quantisation steps, more bits); positive means
            worse quality. Values outside the range are clamped on
            construction.

    Example:
        >>> ROIRegion(x=320, y=180, w=160, h=160, qoffset=-0.5)
        ROIRegion(320,180 160x160 qoffset=-0.500)
    """

    x: int
    y: int
    w: int
    h: int
    qoffset: float

    def __post_init__(self) -> None:
        """Clamp *qoffset* into the supported range."""
        self.qoffset = max(QOFFSET_MIN, min(QOFFSET_MAX, self.qoffset))

    def __repr__(self) -> str:
        return f"ROIRegion({self.x},{self.y} {self.w}x{self.h} qoffset={self.qoffset:.3f})"
