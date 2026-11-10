"""Tests for the backend-agnostic encoder seam."""

import inspect

import numpy as np
import pytest

from ascend_venc_roi import EncoderConfig, EncoderStats, IEncoder, RateControl, ROIRegion


class TestEncoderConfig:
    def test_defaults(self) -> None:
        cfg = EncoderConfig(output_path="out.h265", width=1280, height=720)
        assert cfg.fps == 30.0
        assert cfg.bitrate_kbps == 2000
        assert cfg.rate_control is RateControl.CBR
        assert cfg.gop == 30

    def test_qp_range_default_leaves_roi_headroom(self) -> None:
        # A floor of 24 (the CANN default) would silently clamp ROI requests.
        cfg = EncoderConfig(output_path="out.h265", width=640, height=480)
        assert cfg.qp_min < 24
        assert cfg.qp_max == 51

    def test_ffmpeg_fields_are_gone(self) -> None:
        fields = EncoderConfig.__dataclass_fields__
        for dropped in ("codec", "preset", "crf", "tune", "ffmpeg_path", "input_pix_fmt"):
            assert dropped not in fields


class TestEncoderStats:
    def test_starts_at_zero(self) -> None:
        stats = EncoderStats()
        assert stats.frames_written == 0
        assert stats.frames_dropped == 0
        assert stats.bytes_written == 0
        assert stats.roi_updates == 0


class _FakeEncoder(IEncoder):
    """Minimal in-memory backend, proving IEncoder needs no vendor runtime."""

    def __init__(self) -> None:
        self._open = False
        self._stats = EncoderStats()

    def open(self) -> bool:
        self._open = True
        return True

    def close(self) -> None:
        self._open = False

    def write_frame(
        self,
        frame: np.ndarray,
        regions: list[ROIRegion] | None = None,
    ) -> bool:
        self._stats.frames_written += 1
        if regions:
            self._stats.roi_updates += 1
        return True

    @property
    def stats(self) -> EncoderStats:
        return self._stats

    @property
    def is_open(self) -> bool:
        return self._open


class TestIEncoder:
    def test_context_manager_opens_and_closes(self) -> None:
        enc = _FakeEncoder()
        with enc:
            assert enc.is_open
        assert not enc.is_open

    def test_context_manager_raises_when_open_fails(self) -> None:
        class Failing(_FakeEncoder):
            def open(self) -> bool:
                return False

        with pytest.raises(RuntimeError, match="Failed to open"), Failing():
            pass

    def test_write_frame_counts_roi_updates(self) -> None:
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        with _FakeEncoder() as enc:
            enc.write_frame(frame)
            enc.write_frame(frame, [ROIRegion(0, 0, 16, 16, -0.5)])
        assert enc.stats.frames_written == 2
        assert enc.stats.roi_updates == 1

    def test_no_vendor_type_in_public_signatures(self) -> None:
        """The seam must stay addable-to by a non-Ascend backend."""
        for name in ("open", "close", "write_frame", "stats", "is_open"):
            member = getattr(IEncoder, name)
            func = member.fget if isinstance(member, property) else member
            rendered = str(inspect.signature(func)).lower()
            for vendor_term in ("ascend", "venc", "dvpp", "acl", "himpi", "npu"):
                assert vendor_term not in rendered
