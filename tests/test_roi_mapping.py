"""
Tests for the pure ROI→VENC rectangle arithmetic.

These run on any machine: the module-level helpers do not touch CANN, and
``ascend_venc_roi.ascend_encoder`` imports the ``acl`` bindings lazily.
"""

import pytest

from ascend_venc_roi import ROIRegion
from ascend_venc_roi.ascend_encoder import (
    MAX_ROI_REGIONS,
    map_roi_to_venc_rect,
    qoffset_to_qp,
)

HD = (1280, 720)


class TestQoffsetToQP:
    @pytest.mark.parametrize(
        ("qoffset", "expected_qp"),
        [(-1.0, 10), (-0.5, 19), (0.0, 28), (0.5, 37), (1.0, 46)],
    )
    def test_mapping_table(self, qoffset: float, expected_qp: int) -> None:
        assert qoffset_to_qp(qoffset) == expected_qp

    def test_negative_offset_means_lower_qp(self) -> None:
        # Sign convention: negative qoffset = better quality = lower QP.
        assert qoffset_to_qp(-0.8) < qoffset_to_qp(0.0) < qoffset_to_qp(0.8)

    def test_result_stays_in_h265_range(self) -> None:
        for qoffset in (-10.0, -1.0, 0.0, 1.0, 10.0):
            assert 0 <= qoffset_to_qp(qoffset) <= 51


class TestMapROIToVENCRect:
    def test_position_rounds_down_size_rounds_up(self) -> None:
        result = map_roi_to_venc_rect(ROIRegion(10, 10, 100, 100, 0.0), *HD)
        assert result is not None
        x, y, w, h, _ = result
        assert (x, y) == (0, 0)
        assert (w, h) == (112, 112)

    def test_already_aligned_region_is_unchanged(self) -> None:
        result = map_roi_to_venc_rect(ROIRegion(320, 160, 160, 96, 0.0), *HD)
        assert result == (320, 160, 160, 96, 28)

    @pytest.mark.parametrize("qoffset", [-1.0, -0.25, 0.0, 0.75])
    def test_all_coordinates_are_16_aligned(self, qoffset: float) -> None:
        result = map_roi_to_venc_rect(ROIRegion(37, 91, 133, 217, qoffset), *HD)
        assert result is not None
        x, y, w, h, _ = result
        assert x % 16 == y % 16 == w % 16 == h % 16 == 0

    def test_region_never_leaves_the_frame(self) -> None:
        result = map_roi_to_venc_rect(ROIRegion(1200, 690, 400, 400, 0.0), *HD)
        assert result is not None
        x, y, w, h, _ = result
        assert x + w <= HD[0]
        assert y + h <= HD[1]

    def test_oversized_region_is_clamped_to_frame(self) -> None:
        result = map_roi_to_venc_rect(ROIRegion(0, 0, 9999, 9999, 0.0), *HD)
        assert result is not None
        x, y, w, h, _ = result
        assert (x, y, w, h) == (0, 0, 1280, 720)

    def test_negative_origin_is_clamped_to_zero(self) -> None:
        result = map_roi_to_venc_rect(ROIRegion(-50, -30, 200, 200, 0.0), *HD)
        assert result is not None
        assert result[0] == 0
        assert result[1] == 0

    def test_qp_is_carried_through(self) -> None:
        result = map_roi_to_venc_rect(ROIRegion(64, 64, 64, 64, -0.5), *HD)
        assert result is not None
        assert result[4] == qoffset_to_qp(-0.5)

    def test_frame_too_small_for_a_single_block_yields_none(self) -> None:
        assert map_roi_to_venc_rect(ROIRegion(0, 0, 8, 8, 0.0), 8, 8) is None


def test_hardware_roi_limit_is_eight() -> None:
    assert MAX_ROI_REGIONS == 8
