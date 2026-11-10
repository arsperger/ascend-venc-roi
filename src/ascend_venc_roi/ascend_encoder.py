"""
Ascend VENC hardware encoder with per-frame ROI.

Drives the CANN ``acl.himpi`` VENC API directly to encode BGR frames into an
H.265 elementary stream on an Ascend NPU. Each frame is converted to NV12 on
the VPC (:class:`NV12Converter`) and handed to VENC together with the ROI
attributes for that frame, so the QP map can change on every frame.

This module is the Ascend-specific half of the package. Everything DVPP- or
CANN-shaped is exported from here rather than from the package root, keeping
the root importable and meaningful for a future non-Ascend backend.

Example:
    >>> from ascend_venc_roi import EncoderConfig, ROIRegion
    >>> from ascend_venc_roi.ascend_encoder import AscendVENCEncoder
    >>> cfg = EncoderConfig(output_path="/tmp/out.h265", width=1280, height=720)
    >>> with AscendVENCEncoder(cfg) as enc:  # doctest: +SKIP
    ...     enc.write_frame(frame, [ROIRegion(320, 180, 160, 160, -0.5)])
"""

from __future__ import annotations

import logging
import time
from types import TracebackType
from typing import IO, Any

import numpy as np
import numpy.typing as npt

from ascend_venc_roi.encoder import EncoderConfig, EncoderStats, IEncoder, RateControl
from ascend_venc_roi.nv12_converter import DVPPBuffer, NV12Converter, _align16
from ascend_venc_roi.types import ROIRegion

__all__ = ["AscendVENCEncoder", "DVPPBuffer", "NV12Converter"]

logger = logging.getLogger(__name__)

#: Maximum ROI regions VENC accepts per frame (hardware limit).
MAX_ROI_REGIONS = 8

#: Smallest ROI edge VENC accepts, and the alignment granularity.
_ROI_ALIGN = 16

# qoffset [-1, +1] is mapped onto an absolute QP around this midpoint.
# Negative qoffset means better quality, and lower QP means better quality,
# so the span is *added*, not subtracted.
_QP_BASE = 28
_QP_SPAN = 18
_QP_HARD_MIN = 0
_QP_HARD_MAX = 51

# HiMPI rate-control mode identifiers for H.265.
_RC_MODE_H265_CBR = 11
_RC_MODE_H265_VBR = 12

# Retry budget for venc_query_status after send_frame — the encoder pipeline
# needs a few milliseconds to produce the first packets.
_QUERY_MAX_RETRIES = 50
_QUERY_RETRY_SLEEP_S = 0.001

# Tolerated "already initialised" return codes.
_ACL_ALREADY_INITIALISED = 100002
_HIMPI_SYS_ALREADY_INITIALISED = 0xA0058008


def qoffset_to_qp(qoffset: float) -> int:
    """
    Map a relative quality offset onto an absolute H.265 QP.

    Uses ``QP = 28 + qoffset * 18``, clamped to ``[0, 51]``. The sign is not a
    typo: negative *qoffset* means better quality, and lower QP means better
    quality.

    Args:
        qoffset: Quality offset in ``[-1.0, +1.0]``.

    Returns:
        Absolute quantisation parameter in ``[0, 51]``.

    Example:
        >>> qoffset_to_qp(-0.5), qoffset_to_qp(0.0), qoffset_to_qp(0.5)
        (19, 28, 37)
    """
    qp = int(_QP_BASE + qoffset * _QP_SPAN)
    return max(_QP_HARD_MIN, min(_QP_HARD_MAX, qp))


def map_roi_to_venc_rect(
    roi: ROIRegion,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int, int] | None:
    """
    Clamp and 16-pixel-align an ROI into a rectangle VENC will accept.

    VENC rejects rectangles that are not 16-pixel aligned or that leave the
    frame, so positions are rounded down, sizes rounded up, and the result is
    re-clamped to the frame.

    Args:
        roi: Region to map.
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.

    Returns:
        ``(x, y, width, height, qp)``, or None if nothing at least
        16x16 pixels survives the clamping.

    Example:
        >>> map_roi_to_venc_rect(ROIRegion(10, 10, 100, 100, -0.5), 1280, 720)
        (0, 0, 112, 112, 19)
    """
    x = max(0, min(roi.x, frame_width - _ROI_ALIGN))
    y = max(0, min(roi.y, frame_height - _ROI_ALIGN))
    w = max(_ROI_ALIGN, min(roi.w, frame_width - x))
    h = max(_ROI_ALIGN, min(roi.h, frame_height - y))

    # Positions round down, sizes round up.
    x = x // _ROI_ALIGN * _ROI_ALIGN
    y = y // _ROI_ALIGN * _ROI_ALIGN
    w = _align16(w)
    h = _align16(h)

    # Rounding up may have pushed the rectangle past the frame edge.
    if x + w > frame_width:
        w = (frame_width - x) // _ROI_ALIGN * _ROI_ALIGN
    if y + h > frame_height:
        h = (frame_height - y) // _ROI_ALIGN * _ROI_ALIGN

    if w < _ROI_ALIGN or h < _ROI_ALIGN:
        return None

    return x, y, w, h, qoffset_to_qp(roi.qoffset)


class AscendVENCEncoder(IEncoder):
    """
    Ascend VENC H.265 encoder with per-frame ROI.

    Implements :class:`~ascend_venc_roi.encoder.IEncoder` on the ``acl.himpi``
    VENC API. BGR frames are converted to NV12 on the NPU and encoded to H.265
    with hardware ROI regions that are re-sent for every frame.

    Args:
        config: Encoder configuration.
        venc_chn: VENC channel id.
        vpc_chn: VPC channel id for the internal :class:`NV12Converter`.
        manage_acl: If True, the encoder calls ``acl.init()`` / ``acl.finalize()``
            in :meth:`open` / :meth:`close`. Set False when the surrounding
            application already owns the ACL lifecycle — these calls are
            once-per-process.

    Example:
        >>> cfg = EncoderConfig(
        ...     output_path="/tmp/out.h265", width=1280, height=720,
        ...     bitrate_kbps=4000, gop=30,
        ... )
        >>> with AscendVENCEncoder(cfg) as enc:  # doctest: +SKIP
        ...     for frame, regions in stream:
        ...         enc.write_frame(frame, regions)
    """

    def __init__(
        self,
        config: EncoderConfig,
        *,
        venc_chn: int = 0,
        vpc_chn: int = 0,
        manage_acl: bool = True,
    ) -> None:
        self._config = config
        self._venc_chn = venc_chn
        self._vpc_chn = vpc_chn
        self._manage_acl = manage_acl

        self._converter: NV12Converter | None = None
        self._file: IO[bytes] | None = None
        self._stats = EncoderStats()
        self._frame_idx = 0
        # ROI attrs are channel state; slots left enabled keep applying to later frames.
        self._enabled_roi_slots: set[int] = set()
        self._opened = False
        self._acl_initialised = False
        self._acl_owner = False

        self._w_stride = _align16(config.width)

    # -- IEncoder ---------------------------------------------------------- #

    def open(self) -> bool:
        """
        Initialise the ACL runtime, the VPC converter and the VENC channel.

        Returns:
            True on success. On failure the partial state is torn down and
            the error is logged.
        """
        try:
            if self._manage_acl:
                self._init_acl()
            self._converter = NV12Converter(
                self._config.width, self._config.height, vpc_chn=self._vpc_chn,
            )
            self._converter.open()
            self._create_venc_channel()
            self._file = open(self._config.output_path, "wb")  # noqa: SIM115
            self._opened = True
            self._frame_idx = 0
            self._enabled_roi_slots = set()
            self._stats = EncoderStats()
            logger.info(
                "AscendVENCEncoder opened: %dx%d -> %s (venc_chn=%d)",
                self._config.width, self._config.height,
                self._config.output_path, self._venc_chn,
            )
            return True
        except Exception:
            logger.exception("AscendVENCEncoder.open() failed")
            self.close()
            return False

    def close(self) -> None:
        """Stop VENC, destroy channels, release DVPP memory, finalise ACL."""
        if not self._opened and not self._acl_initialised:
            return

        if self._file is not None:
            self._file.close()
            self._file = None

        self._destroy_venc_channel()

        if self._converter is not None:
            self._converter.close()
            self._converter = None

        if self._manage_acl:
            self._cleanup_acl()
        self._opened = False

        logger.info(
            "AscendVENCEncoder closed: %d frames, %d bytes written",
            self._stats.frames_written, self._stats.bytes_written,
        )

    def write_frame(
        self,
        frame: npt.NDArray[np.uint8],
        regions: list[ROIRegion] | None = None,
    ) -> bool:
        """
        Encode one BGR frame with optional per-frame ROI regions.

        The sequence is: BGR→NV12 on the VPC, ``venc_set_roi_attr`` for up to
        :data:`MAX_ROI_REGIONS` regions, ``venc_send_frame``, then
        ``venc_query_status`` → ``venc_get_stream`` → device-to-host copy →
        ``venc_release_stream``.

        Args:
            frame: BGR frame, ``(H, W, 3)``, dtype ``uint8``.
            regions: ROI regions for this frame. Regions beyond the hardware
                limit of 8 are dropped.

        Returns:
            True if the frame was encoded and written.
        """
        if not self._opened:
            return False

        t0 = time.perf_counter()

        try:
            assert self._converter is not None
            nv12_buf = self._converter.convert(frame)

            self._apply_roi_regions(regions)

            frame_dict = self._build_frame_dict(nv12_buf.addr, self._frame_idx)

            from acl import himpi

            ret: int = himpi.venc_send_frame(self._venc_chn, frame_dict, 5000)
            if ret != 0:
                logger.warning("venc_send_frame failed: 0x%08x (frame %d)", ret, self._frame_idx)
                self._stats.frames_dropped += 1
                self._frame_idx += 1
                return False

            nal_data = self._extract_stream()
            if nal_data is not None and self._file is not None:
                self._file.write(nal_data)
                self._stats.bytes_written += len(nal_data)
                self._stats.frames_written += 1
            else:
                self._stats.frames_dropped += 1

            self._frame_idx += 1
            self._stats.encoding_time_ms += (time.perf_counter() - t0) * 1000
            if self._stats.frames_written > 0:
                self._stats.avg_fps = (
                    self._stats.frames_written / (self._stats.encoding_time_ms / 1000)
                )
            if regions:
                self._stats.roi_updates += 1

            return nal_data is not None

        except Exception:
            logger.exception("write_frame failed at frame %d", self._frame_idx)
            self._stats.frames_dropped += 1
            self._frame_idx += 1
            return False

    @property
    def stats(self) -> EncoderStats:
        """Current encoder statistics."""
        return self._stats

    @property
    def is_open(self) -> bool:
        """Whether the VENC channel is open and accepting frames."""
        return self._opened

    # -- ACL lifecycle ----------------------------------------------------- #

    def _init_acl(self) -> None:
        """Initialise the ACL runtime and the HiMPI subsystem."""
        import acl
        from acl import himpi

        ret: int = acl.init()
        if ret not in (0, _ACL_ALREADY_INITIALISED):
            raise RuntimeError(f"acl.init() failed: {ret}")
        # Only the process that actually initialised ACL may finalise it.
        self._acl_owner = ret == 0

        ret = acl.rt.set_device(0)
        if ret != 0:
            raise RuntimeError(f"set_device(0) failed: {ret}")

        ret = himpi.sys_init()
        if ret not in (0, _HIMPI_SYS_ALREADY_INITIALISED):
            raise RuntimeError(f"sys_init() failed: 0x{ret:08x}")

        self._acl_initialised = True
        logger.debug("ACL runtime initialised (owner=%s)", self._acl_owner)

    def _cleanup_acl(self) -> None:
        """Finalise HiMPI and the ACL runtime, if this encoder owns them."""
        if not self._acl_initialised:
            return
        try:
            import acl
            from acl import himpi

            if self._acl_owner:
                himpi.sys_exit()
                acl.rt.reset_device(0)
                acl.finalize()
        except Exception:
            logger.exception("ACL cleanup error (non-fatal)")
        self._acl_initialised = False

    # -- VENC channel ------------------------------------------------------ #

    def _create_venc_channel(self) -> None:
        """Create and start the VENC channel in H.265 CBR or VBR mode."""
        from acl import himpi

        bitrate_kbps = self._config.bitrate_kbps
        use_vbr = self._config.rate_control is RateControl.VBR

        venc_attr: dict[str, Any] = {
            "type": 265,
            "max_pic_width": self._config.width,
            "max_pic_height": self._config.height,
            "buf_size": self._w_stride * self._config.height * 3 // 2,
            "profile": 0,
            "is_by_frame": 1,
            "pic_width": self._config.width,
            "pic_height": self._config.height,
        }

        rc_attr: dict[str, Any]
        if use_vbr:
            rc_attr = {
                "rc_mode": _RC_MODE_H265_VBR,
                "h265_vbr": {
                    "gop": self._config.gop,
                    "stats_time": 1,
                    "src_frame_rate": int(self._config.fps),
                    "dst_frame_rate": int(self._config.fps),
                    "max_bit_rate": bitrate_kbps,
                },
            }
            rc_label = f"VBR max={bitrate_kbps}"
        else:
            rc_attr = {
                "rc_mode": _RC_MODE_H265_CBR,
                "h265_cbr": {
                    "gop": self._config.gop,
                    "stats_time": 1,
                    "src_frame_rate": int(self._config.fps),
                    "dst_frame_rate": int(self._config.fps),
                    "bit_rate": bitrate_kbps,
                    "max_bit_rate": bitrate_kbps,
                },
            }
            rc_label = f"CBR {bitrate_kbps}"

        chn_attr: dict[str, Any] = {
            "venc_attr": venc_attr,
            "rc_attr": rc_attr,
            # gop_attr is undocumented as mandatory, but omitting normal_p
            # segfaults venc_create_chn.
            "gop_attr": {
                "gop_mode": 0,
                "normal_p": {"ip_qp_delta": 2},
            },
        }

        ret: int = himpi.venc_create_chn(self._venc_chn, chn_attr)
        if ret != 0:
            raise RuntimeError(f"venc_create_chn({self._venc_chn}) failed: 0x{ret:08x}")

        try:
            self._apply_rc_param_qp_range(self._config.qp_min, self._config.qp_max)
        except Exception:
            logger.exception("venc_set_rc_param failed (continuing with CANN defaults)")

        ret = himpi.venc_start_chn(self._venc_chn, {"recv_pic_num": -1})
        if ret != 0:
            himpi.venc_destroy_chn(self._venc_chn)
            raise RuntimeError(f"venc_start_chn({self._venc_chn}) failed: 0x{ret:08x}")

        logger.debug(
            "VENC channel %d started: H265 %s kbps, gop=%d",
            self._venc_chn, rc_label, self._config.gop,
        )

    def _apply_rc_param_qp_range(self, qp_min: int, qp_max: int) -> None:
        """
        Widen the channel's QP range via ``venc_set_rc_param``.

        Performs get → mutate → set, as the CANN documentation recommends,
        rather than constructing a fresh parameter block.

        Args:
            qp_min: New minimum QP for both P/B and I frames.
            qp_max: New maximum QP for both P/B and I frames.

        Raises:
            RuntimeError: If either HiMPI call fails or the parameter block
                has an unexpected shape.
        """
        from acl import himpi

        use_vbr = self._config.rate_control is RateControl.VBR
        mode = _RC_MODE_H265_VBR if use_vbr else _RC_MODE_H265_CBR
        param_key = "h265_vbr_param" if use_vbr else "h265_cbr_param"

        rc_param, ret = himpi.venc_get_rc_param(self._venc_chn, mode)
        if ret != 0:
            raise RuntimeError(
                f"venc_get_rc_param({self._venc_chn}, mode={mode}) failed: 0x{ret:08x}",
            )

        sub = rc_param.get(param_key)
        if not isinstance(sub, dict):
            raise RuntimeError(
                f"rc_param missing expected key '{param_key}'; got {list(rc_param.keys())}",
            )

        old_min, old_max = sub.get("min_qp"), sub.get("max_qp")
        old_min_i, old_max_i = sub.get("min_i_qp"), sub.get("max_i_qp")

        sub["min_qp"] = qp_min
        sub["max_qp"] = qp_max
        sub["min_i_qp"] = qp_min
        sub["max_i_qp"] = qp_max

        ret = himpi.venc_set_rc_param(self._venc_chn, rc_param)
        if ret != 0:
            raise RuntimeError(f"venc_set_rc_param({self._venc_chn}) failed: 0x{ret:08x}")

        logger.info(
            "VENC chn %d %s QP range: P/B [%s,%s]->[%d,%d], I [%s,%s]->[%d,%d]",
            self._venc_chn, param_key,
            old_min, old_max, qp_min, qp_max,
            old_min_i, old_max_i, qp_min, qp_max,
        )

    def _destroy_venc_channel(self) -> None:
        """Stop and destroy the VENC channel, tolerating partial init."""
        try:
            from acl import himpi

            himpi.venc_stop_chn(self._venc_chn)
            himpi.venc_destroy_chn(self._venc_chn)
        except Exception:
            logger.debug("VENC channel destroy error (may be already destroyed)")

    # -- ROI --------------------------------------------------------------- #

    def _apply_roi_regions(self, regions: list[ROIRegion] | None) -> None:
        """Set this frame's ROI slots and disable every slot it does not use."""
        if not regions and not self._enabled_roi_slots:
            return

        from acl import himpi

        set_now: set[int] = set()
        for idx, roi in enumerate((regions or [])[:MAX_ROI_REGIONS]):
            mapped = map_roi_to_venc_rect(roi, self._config.width, self._config.height)
            if mapped is None:
                continue
            x, y, w, h, qp = mapped

            roi_attr: dict[str, Any] = {
                "idx": idx,
                "enable": 1,
                "is_abs_qp": 1,
                "qp": qp,
                # Keys are x/y/width/height — not left/top.
                "rect": {"x": x, "y": y, "width": w, "height": h},
            }
            himpi.venc_set_roi_attr(self._venc_chn, roi_attr)
            set_now.add(idx)

            if self._frame_idx <= 1 and idx < 3:
                logger.debug(
                    "ROI[%d] qoffset=%.3f QP=%d rect=(%d,%d,%dx%d)",
                    idx, roi.qoffset, qp, x, y, w, h,
                )

        for idx in sorted(self._enabled_roi_slots - set_now):
            himpi.venc_set_roi_attr(self._venc_chn, {
                "idx": idx,
                "enable": 0,
                "is_abs_qp": 1,
                "qp": _QP_BASE,
                "rect": {"x": 0, "y": 0, "width": _ROI_ALIGN, "height": _ROI_ALIGN},
            })
        self._enabled_roi_slots = set_now

    # -- frame submission -------------------------------------------------- #

    def _build_frame_dict(self, dev_addr: int, frame_idx: int) -> dict[str, Any]:
        """Build the VENC frame descriptor for an NV12 buffer at *dev_addr*."""
        y_size = self._w_stride * self._config.height

        v_frame: dict[str, Any] = {
            "width": self._config.width,
            "height": self._config.height,
            "field": 1,  # HI_VIDEO_FIELD_FRAME
            "pixel_format": 1,  # NV12
            "video_format": 0,
            "compress_mode": 0,
            "dynamic_range": 0,
            "color_gamut": 1,  # BT.709
            "header_stride": [0, 0, 0],
            "width_stride": [self._w_stride, self._w_stride, 0],
            "height_stride": [0, 0, 0],
            "header_phys_addr": [0, 0, 0],
            "phys_addr": [0, 0, 0],
            "header_virt_addr": [0, 0, 0],
            "virt_addr": [dev_addr, dev_addr + y_size, 0],
            # Must be even and strictly increasing. A constant value (0 for
            # every frame) produces alternating frame corruption.
            "time_ref": (frame_idx + 1) * 2,
            "pts": frame_idx * 33333,
            "user_data": [0],
            "frame_flag": 0,
            "supplement": {
                "jpeg_dcf_phy_addr": 0, "jpeg_dcf_vir_addr": 0,
                "isp_frame_phy_addr": 0, "isp_frame_vir_addr": 0,
                "metadata_phy_addr": 0, "metadata_vir_addr": 0,
                "jpeg_dcf_size": 0, "isp_frame_size": 0,
                "metadata_size": 0, "frame_dng_phy_addr": 0,
                "frame_dng_vir_addr": 0, "frame_dng_size": 0,
                "low_delay_phy_addr": 0, "low_delay_vir_addr": 0,
            },
        }
        return {"v_frame": v_frame, "pool_id": 0xFFFFFFFF, "mod_id": 0}

    def _extract_stream(self) -> bytes | None:
        """Poll for encoded packets, copy them to host memory, release them."""
        import acl
        from acl import himpi

        status = None
        for _ in range(_QUERY_MAX_RETRIES):
            status, ret = himpi.venc_query_status(self._venc_chn)
            if ret == 0 and status.get("cur_packs", 0) > 0:
                break
            time.sleep(_QUERY_RETRY_SLEEP_S)
        else:
            logger.debug("No packs ready after %d retries", _QUERY_MAX_RETRIES)
            return None

        stream, ret = himpi.venc_get_stream(
            self._venc_chn, {"pack_cnt": status["cur_packs"]}, 5000,
        )
        if ret != 0:
            logger.warning("venc_get_stream failed: 0x%08x", ret)
            return None

        try:
            packs = stream.get("pack", [])
            if not isinstance(packs, list):
                packs = [packs]

            nal_chunks: list[bytes] = []
            for p in packs:
                if not isinstance(p, dict):
                    continue
                addr = p.get("addr", 0)
                total_len = p.get("len", 0)
                offset = p.get("offset", 0)
                if addr and total_len > offset:
                    data_len = total_len - offset
                    buf = np.zeros(data_len, dtype=np.uint8)
                    copy_ret: int = acl.rt.memcpy(
                        buf.ctypes.data, data_len, addr + offset, data_len, 2,
                    )  # 2 = device to host
                    if copy_ret == 0:
                        nal_chunks.append(buf.tobytes())

            return b"".join(nal_chunks) if nal_chunks else None
        finally:
            himpi.venc_release_stream(self._venc_chn, stream)

    # -- context manager --------------------------------------------------- #

    def __enter__(self) -> AscendVENCEncoder:
        """Open the encoder, raising if it cannot be opened."""
        if not self.open():
            raise RuntimeError("Failed to open AscendVENCEncoder")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the encoder."""
        self.close()
