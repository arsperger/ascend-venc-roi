"""
ascend-venc-roi — per-frame region-of-interest H.265 encoding on Ascend NPUs.

The package root is deliberately backend-neutral: it exports the encoder
interface, its configuration and the ROI primitives, and nothing that names a
vendor. The Ascend implementation lives one import away::

    from ascend_venc_roi import EncoderConfig, ROIRegion, RateControl
    from ascend_venc_roi.ascend_encoder import AscendVENCEncoder

    cfg = EncoderConfig(output_path="out.h265", width=1280, height=720)
    with AscendVENCEncoder(cfg) as enc:
        for frame, boxes in stream:
            enc.write_frame(frame, [ROIRegion(*b, qoffset=-0.5) for b in boxes])

Importing this module does not require CANN — the ``acl`` bindings are
imported lazily, when a backend is actually opened.
"""

from ascend_venc_roi.encoder import (
    EncoderConfig,
    EncoderStats,
    IEncoder,
    RateControl,
)
from ascend_venc_roi.types import QOFFSET_MAX, QOFFSET_MIN, ROIRegion

__version__ = "0.1.0"

__all__ = [
    "QOFFSET_MAX",
    "QOFFSET_MIN",
    "EncoderConfig",
    "EncoderStats",
    "IEncoder",
    "ROIRegion",
    "RateControl",
    "__version__",
]
